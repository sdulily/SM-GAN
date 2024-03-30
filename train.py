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
from collections import OrderedDict
import numpy as np
from util import html, util
from util import visualizer
import os
import torch
from options.train_options import TrainOptions
from dataset.aligned_dataset import AlignedDataset
from models.tim_gan import LocalGAN

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()
  if not os.path.exists(os.path.join(opts.output_dir, 'model', opts.name)):
    os.makedirs(os.path.join(opts.output_dir, 'model', opts.name))

  # data loader
  print('\n--- config dataset ---')
  dataset = AlignedDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)#一开始这个shuffle是true 以为是每组打乱的，结果最后不是一组一组打乱的，而是A和B对不上号了，就什么都不会学到

  # for it, (src_img, text, tgt_img) in enumerate(dataset):
  #   print(it)
  #   print(src_img.shape)
  #   print(text)
  #   print(tgt_img.shape)
  # exit(0)


  # model
  print('\n--- create model ---')
  model = LocalGAN(opts)
  model.cuda()

  # training
  print('\n--- training ---')
  total_it = 0

  for ep in range(opts.n_ep):
    for it, (src_img, text, tgt_img) in enumerate(loader):
      # print(it)
      # print(src_img.shape)
      # print(text)
      # print(tgt_img.shape)
      # exit(0)
      # forward
      temperature_rate = max(0, 1 - (ep + 1)/float(opts.n_ep))
      use_gt_attn_rate = max(0, 1 - ep/float(opts.n_ep))
      model.set_input(src_img, text, tgt_img)
      model.forward(use_gt_attn_rate=use_gt_attn_rate,temperature_rate=temperature_rate)

      if it%200==0:
        model.save_img(opts.batch_size,ep,it)

      # update
      model.update_D()
      model.update_G()

      # display
      model.write_display(total_it)


      # print
      if (it + 1) % (len(loader) // 5) == 0:
        print('Iteration {}, EP[{}/{}]'.format(total_it + 1, ep + 1, opts.n_ep))
      total_it += 1


    # write model file
    if (ep + 1) % 5 == 0:#本来是余10
      model.save(os.path.join(opts.output_dir, 'model', opts.name, '{}.pth'.format(ep + 1)), ep, total_it)

  return

if __name__ == '__main__':
  main()
