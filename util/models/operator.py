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
import torch
import torch.nn as nn
import torch
from third_party import networks
import torch.nn.functional as F
from torch.autograd import Variable
from options.train_options import TrainOptions

class GLU(nn.Module):
  def __init__(self):
    super(GLU, self).__init__()

  def forward(self, x):
    nc = x.size(1)
    assert nc % 2 == 0, 'channels dont divide 2!'
    nc = int(nc / 2)
    return x[:, :nc] * F.sigmoid(x[:, nc:])

class CA_NET(nn.Module):
  def __init__(self):
    super(CA_NET, self).__init__()
    self.t_dim = 512#256
    #self.t_dim = cfg.TEXT.EMBEDDING_DIM 256
    self.c_dim = 100
    #self.c_dim = cfg.GAN.CONDITION_DIM
    self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
    self.relu = GLU()

  def encode(self, text_embedding):
    a=self.fc(text_embedding)#2*400
    x = self.relu(a)#2*200
    mu = x[:, :self.c_dim]#大小2*100
    logvar = x[:, self.c_dim:]#2*100
    return mu, logvar

  def reparametrize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()

    eps = torch.cuda.FloatTensor(std.size()).normal_()# 在cuda中生成一个std.size()的张量，标准正态分布采样，类型为FloatTensor

    eps = Variable(eps)#Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
    return eps.mul(std).add_(mu)

  def forward(self, text_embedding):
    mu, logvar = self.encode(text_embedding)#编码  debug的时候这个地方的输出就变成了2*100 两个都是
    c_code = self.reparametrize(mu, logvar)#重新参数化成正态分布
    return c_code, mu, logvar

# The implementation of ACM (affine combination module)

class ACM(nn.Module):
    def __init__(self, channel_num):
        super(ACM, self).__init__()#channel_num输入的是32
        self.conv = conv3x3(256, 128)#将通道数变成128 本来的timgan的通道数好像是3吧
        self.conv_weight = conv3x3(128, channel_num)    # weight
        self.conv_bias = conv3x3(128, channel_num)      # bias

        self.conv2d = torch.nn.Conv2d(in_channels=32, out_channels=256, kernel_size=(1, 1), stride=1, padding=0).cuda()

    def forward(self, x, img):#这里的x是h_code
        out_code = self.conv(img)
        out_code_weight = self.conv_weight(out_code)
        out_code_bias = self.conv_bias(out_code)
        edit_feat=x * out_code_weight + out_code_bias#对应位置处的两个元素相乘
        edit_feature=self.conv2d(edit_feat)#将2*32*64*64  转化为2*256*64*64
        return edit_feature


def conv3x3(in_planes, out_planes):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                   padding=1, bias=False)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),#后面的上采样直接两倍 64*2=128 128*2=256
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU())
    return block



class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):#初始化的时候是512和100
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf#512
        self.in_dim = 100 + ncf+256#200

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim#456 512
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),#456 16384
            nn.BatchNorm1d(ngf * 4 * 4 * 2),#16384
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)


        self.max_pool2d= nn.MaxPool2d(kernel_size=3, stride=2)  # 最大池化2*256*31*31
        self.conv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=1,
                               padding=0)  # 卷积2*1*31*31
        self.fc1=nn.Linear(961, 256)

    def forward(self, z_code, c_code,isTrain,real_A_feat):#2*100 2*512  这里是根据real_a_feat才能生成cnn_code
#debug的时候说c_code是2*100 z_code也是2*100
        c_z_code = torch.cat((c_code, z_code), 1)#2*200
#debug的时候是2*200


        cnn_code1 = self.max_pool2d(real_A_feat)  # 池化2*256*31*31
        cnn_code2 = self.conv(cnn_code1)# 卷积2*1*31*31
        cnn_code3 = cnn_code2.view(cnn_code2.size(0), -1)  # 2,31*31
        cnn_code = self.fc1(cnn_code3)  # 2*256

# for testing
        if not isTrain:#测试的时候是重复
            cnn_code = cnn_code.repeat(c_z_code.size(0), 1)

        c_z_cnn_code = torch.cat((c_z_code, cnn_code), 1) #2*（200+256）

        # state size ngf x 4 x 4
        out_code = self.fc(c_z_cnn_code)#2*16384
        out_code = out_code.view(-1, self.gf_dim, 4, 4)#2*512*4*4
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)#bs*32*64*64

        return out_code64




#-----------------------------------------------------



class Adaptive_Routing(torch.nn.Module):
  def __init__(self, n_res, dim, text_dim, temperature=1.):
    # super(AdaIN, self).__init__()
    super(Adaptive_Routing, self).__init__()
    #layer1
    self.res = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')#构建n_res层resblocks
    self.mlp = networks.MLP(text_dim, self.get_num_adain_params(self.res), text_dim, 3, norm='none', activ='relu')
    self.res2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp2 = networks.MLP(text_dim, self.get_num_adain_params(self.res2), text_dim, 3, norm='none', activ='relu')
    self.res3 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp3 = networks.MLP(text_dim, self.get_num_adain_params(self.res3), text_dim, 3, norm='none', activ='relu')
    #layer2
    self.res_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res_2), text_dim, 3, norm='none', activ='relu')
    self.res2_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp2_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res2_2), text_dim, 3, norm='none', activ='relu')
    self.res3_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp3_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res3_2), text_dim, 3, norm='none', activ='relu')
    self.mlpencoder = networks.MLP(text_dim, 3*2, text_dim, 3, norm='none', activ='relu')
    #Temperature
    self.T = temperature


  def forward(self, img_feat, text_feat, temperature_rate):
    gates = self.mlpencoder(text_feat.detach())
    gates = gates.view(-1,2,3)
    gates = F.gumbel_softmax(gates, self.T, hard=False)
    gates = gates.view(-1,2,3,1,1,1)

    adain_params = self.mlp(text_feat)#[batchsize,参数量]
    self.assign_adain_params(adain_params, self.res)#将参数传入adain1-1层
    out1 = self.res(img_feat)#将参数传入这个块 实际上代码中adain模块是在resblock中的，论文中为了形象的表示传参数 才分开画

    adain_params2 = self.mlp2(text_feat)
    self.assign_adain_params(adain_params2, self.res2)
    out2 = self.res2(img_feat)

    adain_params3 = self.mlp3(text_feat)
    self.assign_adain_params(adain_params3, self.res3)
    out3 = self.res3(img_feat)

    feat = out1*gates[:,0,0]+out2*gates[:,0,1]+out3*gates[:,0,2]

    adain_params_2 = self.mlp_2(text_feat)
    self.assign_adain_params(adain_params_2, self.res_2)
    out1 = self.res_2(feat)

    adain_params2_2 = self.mlp2_2(text_feat)
    self.assign_adain_params(adain_params2_2, self.res2_2)
    out2 = self.res2_2(feat)

    adain_params3_2 = self.mlp3_2(text_feat)
    self.assign_adain_params(adain_params3_2, self.res3_2)
    out3 = self.res3_2(feat)

    return out1*gates[:,1,0]+out2*gates[:,1,1]+out3*gates[:,1,2], gates

  def assign_adain_params(self, adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        mean = adain_params[:, :m.num_features]
        std = adain_params[:, m.num_features:2*m.num_features]
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*m.num_features:
          adain_params = adain_params[:, 2*m.num_features:]

  def get_num_adain_params(self, model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        num_adain_params += 2*m.num_features#一维向量4096个数
    return num_adain_params
