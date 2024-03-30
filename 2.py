import PIL
import numpy as np
import torch
# from torchvision import transforms
# from PIL import Image
# dataroot=r'F:\lrc\tim-gan-main\tim-gan-main\output\TIMGAN\clevr_test_60_self\images\0_input.jpg'
# image = Image.open(dataroot)
# to_tensor = transforms.ToTensor()
# img_tensor = to_tensor(image)    # img_tensor的每个通道最大值为1.0，最小值为0
# img=torch.sigmoid(img_tensor)
# img=transforms.ToPILImage()(img)
# img.show()
#
# image names

from os.path import join
import numpy as np
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random
# from dataset import util
# images_A = util.image_file(os.listdir(join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_A')))
# images_A.sort()
# images_A = [join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_A', name) for name in images_A]
# images_B = util.image_file(os.listdir(join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_B')))
# images_B.sort()
# images_B = [join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_B', name) for name in images_B]
# 将图片放到正确的文件夹下而已 分成_A和_B
# text 这里的处理方式应该变化 变成和图片一样的
# self.texts = np.load(os.path.join(opts.dataroot,'%s_text.npy'%opts.phase))#np.load('../dataset/Clevr/train_text.npy')
# texts = util.text_file(os.listdir(join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_text')))
# texts.sort()
# texts = [join(r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr', 'test' + '_text', name) for name in texts]
# print(texts)
# 这是获取了所有的文件地址+文件名 类似于../dataset/Clevr/train_text\CLEVR_new_000004_1.txt
# text
# images

# def load_image(filename):
#     img = Image.open(filename).convert('RGB')
# src_img = load_image(images_A[8])  # images_A[index]返回的是序号index的图片 比如index=1 ../dataset/Clevr/train_A\CLEVR_source_000001_1.png
# tgt_img = load_image(images_B[8])
#
# A_path = images_A[8]  # 返回某一张图片的路径 比如../dataset/Clevr/train_A\CLEVR_source_000035_1.png
#
# if 'Clevr' in r'F:\lrc\tim-gan-main\tim-gan-main\dataset\Clevr':
#     text_idx = int(A_path[-12:-6])  # 获取的是000035 返回的就是35 int类型
# else:
#     text_idx = int(A_path[A_path.rfind('/') + 1:A_path.rfind('.')])
#
#
# textpath = texts[text_idx]#这个地方也是返回的一个文本的路径  训练集10344张 测试集中编号没改 这里减去之后还是从0开始
#     # print(textpath)# 输出CLEVR_new_000000_1.txt 而不是完整的路径
#
# f = open(textpath, encoding="utf-8")
#     # 输出读取到的数据
# text = f.read()
# textpath = texts[text_idx]  # 这个地方也是返回的一个文本的路径  训练集10344张 测试集中编号没改 这里减去之后还是从0开始
# print(textpath)# 输出CLEVR_new_000000_1.txt 而不是完整的路径

#
# import torchvision.utils as vutils
#
# x_fake也是tensor pathjoin就是路径  重点是normalize是true 最后一个参数第[0]维
# vutils.save_image(x_fake,os.path.join(args.res_dir, '{}_EMA_{}_{}.jpg'.format(args.gpu, epoch, count)),
#                               normalize=True, nrow=args.batch_size)
