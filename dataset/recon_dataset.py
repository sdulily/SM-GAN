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
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random
from dataset import util

class ReconDataset(data.Dataset):

  def __init__(self, opts):
    self.dataroot = opts.dataroot
    # print(self.dataroot)
    self.train = (opts.phase == 'train')#self.train=true
    self.resize_ratio = opts.resize_ratio
    self.crop_size = opts.crop_size


    # image names dataroot是'dataset/Clevr/'
    names1 = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_A')))#listdir是返回指定文件夹包含的文件或文件夹名字的列表。该列表顺序以字母排序
    names1.sort()
    names2 = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_B')))  # listdir是返回指定文件夹包含的文件或文件夹名字的列表。该列表顺序以字母排序
    names2.sort()

    # image
    if 'CSSDataset' in self.dataroot:
      self.images = [join(self.dataroot, opts.phase + '_A', name) for name in names1] +\
                    [join(self.dataroot, opts.phase + '_B', name.replace('source', 'target')) for name in names2]#B是source是旧的 （old,new）
    else:
      self.images = [join(self.dataroot, opts.phase + '_A', name) for name in names1] +\
                    [join(self.dataroot, opts.phase + '_B', name) for name in names2]
    # if 'CSSDataset' in self.dataroot:本来是用的下面被注释掉的部分的
    #   self.images = [join(self.dataroot, opts.phase + '_A', name) for name in names] +\
    #                 [join(self.dataroot, opts.phase + '_B', name.replace('source', 'target')) for name in names]#B是source是旧的 （old,new）
    # else:
    #   self.images = [join(self.dataroot, opts.phase + '_A', name) for name in names]

    self.input_dim = opts.input_dim#3
    self.flip = opts.flip#true
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))#其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
    self.transforms = Compose(transforms)
    
    self.dataset_size = len(self.images)
    print('{} recon dataset size {}'.format(opts.phase, self.dataset_size))
    return

  def __getitem__(self, index):

    # read image
    img = Image.open(self.images[index]).convert('RGB')# Image是PIL库中的 这里的images应该是个列表

    # flip翻转
    flip = random.randint(0, 1) if self.flip and self.train else 0
    if flip == 1:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # resize调整大小
    resize_ratio = random.uniform(1, self.resize_ratio) if self.train else 1
    resize_size = (int(self.crop_size[0]*resize_ratio), int(self.crop_size[1]*resize_ratio))
    img = F.resize(img, resize_size, Image.BICUBIC)
    #resize_size = random.randint(self.crop_size, int(self.crop_size*self.resize_ratio)) if self.train else self.crop_size
    #img = F.resize(img, (resize_size, resize_size), Image.BICUBIC)

    # crop裁剪
    if self.train:
      crop = util.get_crop_params(resize_size, self.crop_size)
      img = F.crop(img, crop[0], crop[1], crop[2], crop[3])#cropsize可以调整
    else:
      img = F.center_crop(img, self.crop_size)
      #img = F.center_crop(img, (self.crop_size, self.crop_size))

    # transform
    img = self.transforms(img)

    # dimension stuff
    if self.input_dim == 1:#默认是3
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)

    return img

  def __len__(self):
    return self.dataset_size
