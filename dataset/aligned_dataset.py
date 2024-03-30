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
from os.path import join
import numpy as np
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random
from dataset import util

class AlignedDataset(data.Dataset):

  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.train = (opts.phase == 'train')#true
    self.resize_ratio = opts.resize_ratio#1.1
    self.crop_size = opts.crop_size#是根据坐标进行裁剪的 但是可以自定义

    # image names
    self.images_A = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_A')))
    self.images_A.sort()
    self.images_A = [join(self.dataroot, opts.phase + '_A', name) for name in self.images_A]
    self.images_B = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_B')))
    self.images_B.sort()
    self.images_B = [join(self.dataroot, opts.phase + '_B', name) for name in self.images_B]
    #将图片放到正确的文件夹下而已 分成_A和_B
    # text 这里的处理方式应该变化 变成和图片一样的
    # self.texts = np.load(os.path.join(opts.dataroot,'%s_text.npy'%opts.phase))#np.load('../dataset/Clevr/train_text.npy')
    self.texts = util.text_file(os.listdir(join(self.dataroot, opts.phase + '_text')))
    self.texts.sort()
    self.texts = [join(self.dataroot, opts.phase + '_text', name) for name in self.texts]
    # print(texts)
    #这是获取了所有的文件地址+文件名 类似于../dataset/Clevr/train_text\CLEVR_new_000004_1.txt

    # image transformation
    self.input_dim = opts.input_dim#3
    self.flip = opts.flip#true
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))##其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
    self.transforms = Compose(transforms)
    
    self.dataset_size = len(self.images_B)
    print('{} aligned dataset size {}'.format(opts.phase, self.dataset_size))
    return

  def load_image(self, filename, flip, resize_size, crop):
    img = Image.open(filename).convert('RGB')
    
    # flip
    if flip == 1:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # resize
    img = F.resize(img, resize_size, Image.BICUBIC)

    # crop
    if self.train:
      img = F.crop(img, crop[0], crop[1], crop[2], crop[3])
    else:
      img = F.center_crop(img, self.crop_size)

    # transform
    img = self.transforms(img)

    # dimension stuff
    if self.input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    
    return img

  def __getitem__(self, index):
    
    # image augmentation数据增强
    flip = random.randint(0, 1) if self.flip and self.train else 0
    resize_ratio = random.uniform(1, self.resize_ratio) if self.train else 1
    resize_size = (int(self.crop_size[0]*resize_ratio), int(self.crop_size[1]*resize_ratio))
    crop = util.get_crop_params(resize_size, self.crop_size)

    # images
    src_img = self.load_image(self.images_A[index], flip, resize_size, crop)#images_A[index]返回的是序号index的图片 比如index=1 ../dataset/Clevr/train_A\CLEVR_source_000001_1.png
    tgt_img = self.load_image(self.images_B[index], flip, resize_size, crop)

    # text
    A_path = self.images_A[index]#返回某一张图片的路径 比如../dataset/Clevr/train_A\CLEVR_source_000035_1.png
    if 'clevr' in self.dataroot:
      text_idx = int(A_path[-12:-6])# 获取的是000035 返回的就是35 int类型
    else:
      text_idx = int(A_path[A_path.rfind('/')+1:A_path.rfind('.')])

    textpath = self.texts[text_idx]#这个地方也是返回的一个文本的路径
    # print(textpath)# 输出CLEVR_new_000000_1.txt 而不是完整的路径

    f = open(textpath, encoding="utf-8")
    # 输出读取到的数据
    text = f.read()
    # 读
    # 分隔开split 自动返回列表list类型
    # text=text.split(' ')
    # text.extend(["null"]*(11-len(text)))

    return src_img, text, tgt_img#如果最终返回的是一句话 那就和图片的处理格式变成一样的

  def __len__(self):
    return self.dataset_size
