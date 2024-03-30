# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models for Text and Image Composition."""

import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class BertTextEncoder(torch.nn.Module):
  """Base class for image + text composition."""

  def __init__(self, pretrained=True, img_dim=256):
    super(BertTextEncoder, self).__init__()
    self.tokenizer = BertTokenizer.from_pretrained("models/bert-base-cased/")
    self.pretrained = pretrained
    ### Define an attention module using concat and additive 使用concat和additive来定义一个注意力模块
    self.query1 = torch.nn.Linear(768+img_dim, 512)#input=1024 output=512 weight=【512，1024】因为矩阵相乘中权重求的是一个转置 偏置bias=512 实际上就是一个矩阵的转置乘法 默认bias=true参加训练
    self.key1 = torch.nn.Linear(768, 512)
    self.value1 = torch.nn.Linear(768, 512)
    self.query2 = torch.nn.Linear(768+img_dim, 512)
    self.key2 = torch.nn.Linear(768, 512)
    self.value2 = torch.nn.Linear(768, 512)
    if not pretrained:
      config = BertConfig.from_pretrained("models/bert-base-cased/")
      config.hidden_size = 768
      config.num_attention_heads = 12
      self.textmodel = BertModel(config)
    else:
      self.textmodel = BertModel.from_pretrained("models/bert-base-cased/")
      #self.downsample = torch.nn.Linear(768,512)

  def extract_text_feature(self, texts, img1d):
    x = []
    xlen = []
    mask = []
    attmask = []
    text_tokens = []
    for text in texts:#这个texts里面是batchsize句文本，每次处理一句话
      t = '[CLS] '+str(text[0].upper())+str(text[1:])+' [SEP]'#这个t是分割成单句话，并且将单句话的首字母大写 并且在句子的前后加上了两个标识符
      tokenized_text = self.tokenizer.tokenize(t)#将一句话token化
      #print(tokenized_text) # ['[CLS]', 'Change', 'the', 'small', 'c', '##yan', 'metal', 'cube', 'into', 'small', 'brown', 'metal', 'sphere', '[SEP]']
      text_tokens.append(tokenized_text) #将这句话的token加进去 最后会形成batchsize句token的合集
      indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)#这里是将所有token化的所有词 找到一个对应的int类型的整数，如果后面还有embedding 这个的作用是将int类型的这个变成了词向量的形式
      #print(indexed_tokens) # [101, 9091, 1103, 1353, 3058, 9579, 7613, 1154, 1353, 2448, 9579, 11036, 102]
      x.append(indexed_tokens)#将这个整数类型句子token的加到x里
      xlen.append(len(indexed_tokens))
    maxlen = max(xlen)
    for i in range(len(x)):#开始对于句子长短不一的补0
      mask.append([0]+[1]*(xlen[i]-2)+[0]*(maxlen-xlen[i]+1))#这里减去2是因为句子的开头和结尾都有两个标识符 去掉
      attmask.append([1]*(xlen[i])+[0]*(maxlen-xlen[i]))#这两个mask attmask都是mask mask指的是不关注的那些地方 不需要关注的地方
      x[i] = x[i]+[0]*(maxlen-xlen[i])
    x = torch.tensor(x)
    mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2)#在第三个维度增加一个维度
    attmask = torch.tensor(attmask, dtype=torch.float).unsqueeze(2)
    itexts = torch.autograd.Variable(x).cuda()
    mask = torch.autograd.Variable(mask).cuda()
    attmask = torch.autograd.Variable(attmask).cuda()#在PyTorch0.4.0之后Variable 已经被PyTroch弃用，Variable不再是张量使用autograd的必要条件，只需要将张量的requires_grad设为True该张量就会自动支持autograd运算
    out = self.textmodel(itexts, attention_mask = attmask)#这里textmodel才是bert模型的入口 进去的是句子中单词变成对应的整数的形式 [101, 9091, 1103, 1353, 3058, 9579, 7613, 1154, 1353, 2448, 9579, 11036, 102]
    xlen = (torch.tensor(xlen, dtype = torch.float)-2).view(-1,1).data.cuda()#remove special token 这里减去2就是要去掉单个句子开头和结尾的两个特殊标识符
    assert tuple(out[0].shape) == (x.size()[0], maxlen, self.textmodel.config.hidden_size)#out[0] 如果batchsize为16的话 那就是输出16个句子 句意
    #out[0]本来是标识符[cls] 但是经过bert 开头的这个标识符会根据句子的意思进行不同的变换 可以意思为out[0]变成了这句话的句意
    #这个== 意思是batchsize中句子的数量 要一致
    
    puretext = torch.div(torch.sum(torch.mul(out[0],mask), dim=1),xlen)#div是张量和标量做逐元素除法 mul是逐元素相乘
    #这个puretext就是很纯净的文本 使用mask除掉那些不应该被关注的0
    comb = torch.cat((puretext, img1d),dim=1).unsqueeze(1)#img1d是real_A的特征 在第二个维度增加一个维度
    masked_out = torch.mul(out[0],mask)
    mask_sfmax = (1-mask)*-10000.0

    #这里分成两拨是 模型中的 都进如attention 来获取 where how
    query1 = self.query1(comb)
    key1 = self.key1(masked_out)
    value1 = self.value1(masked_out)
    logit1 = torch.sum(query1*key1,dim=2)/math.sqrt(512)+mask_sfmax[:,:,0]
    attn1 = torch.nn.functional.softmax(logit1,dim=1)
    out1 = torch.sum(attn1.unsqueeze(2)*value1,dim=1)

    query2 = self.query2(comb)
    key2 = self.key2(masked_out)
    value2 = self.value2(masked_out)
    logit2 = torch.sum(query2*key2,dim=2)/math.sqrt(512)+mask_sfmax[:,:,0]
    attn2 = torch.nn.functional.softmax(logit2,dim=1)
    out2 = torch.sum(attn2.unsqueeze(2)*value2,dim=1)

    rawtext = torch.cat((puretext, torch.zeros((puretext.shape[0],1024-768),device=puretext.device)),dim=1)
    return (out1, attn1), (out2, attn2), text_tokens,  rawtext
  #text1=（out1，attn1） text1和text2为什么可以分别进相同的selfattention 但是对于不同位置的权重不一样呢 一个是where关键词的权重高，一个是how关键词的权重高
  #是不是selfattention会自动学习qkv的线性矩阵，自己学习到了不同的权重  出来的时候已经同样位置的词是不一样的权重了
  