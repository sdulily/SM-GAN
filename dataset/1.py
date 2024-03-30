import os

import numpy as np

from dataset import util

from os.path import join
# str='a'
# str2='../datasets/Clevr/'
# list=join(str2,str)                                    #../datasets/Clevr/a
# names =join('../datasets/Clevr/', 'train' + '_A')      #../datasets/Clevr/train_A
# print(list)
# print(names)

from os.path import join
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random
from dataset import util


# a = ('train' == 'train')
# print(a)#true

from os.path import join
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random

from dataset import util
# names = util.image_file(os.listdir(join('../dataset/Clevr/', 'train' + '_A')))#listdir是返回指定文件夹包含的文件或文件夹名字的列表。该列表顺序以字母排序
# names.sort()
# images = [join('../dataset/Clevr/', 'train' + '_A', name) for name in names] +\
#                     [join('../dataset/Clevr/', 'train' + '_B', name.replace('source', 'target')) for name in names]
#

# images_A = util.image_file(os.listdir(join('../dataset/Clevr/', 'train' + '_A')))
# images_A.sort()
# images_A = [join('../dataset/Clevr/', 'train' + '_A', name) for name in images_A]
# print(images_A[0])
# print(images_A[1])
# print(images_A[2])
# print(images_A[3])
#结果
# ../dataset/Clevr/train_A\CLEVR_source_000000_0.png
# ../dataset/Clevr/train_A\CLEVR_source_000001_0.png
# ../dataset/Clevr/train_A\CLEVR_source_000002_0.png
# ../dataset/Clevr/train_A\CLEVR_source_000003_0.png



#这段代码是将 _0和_1的图片分开到不同的文件夹下 然后执行rename.py下的代码！！！
import os
import shutil

path = r'./Clevr/images_test_3_add'
AData = r'./Clevr/test_A'
BData = r'./Clevr/test_B'
files_list = os.listdir(path)#./是当前目录

for file in files_list:
    filename, suffix = os.path.splitext(file)  # filename是文件名 suffix是文件后缀
    file_path = path + "/" + filename +'.png'
    # print(filename)
    label1 = filename.split('_')[3]  # 最后一个'_'后面是1 还是0
    if label1 == '0':
        shutil.copy(file_path, BData)
    # else:
    #     shutil.copy(file_path, AData)
    elif label1 =='3':
        shutil.copy(file_path, AData)



# texts = os.path.join('../dataset/Clevr/','%s_text.npy'%'train')
# print(texts)
# #结果是 ../dataset/Clevr/train_text.npy


# A_path = '../dataset/Clevr/train_A\CLEVR_source_000035_1.png'#返回某一张图片的路径 比如../dataset/Clevr/train_A\CLEVR_source_000001_1.png
# text_idx = int(A_path[-12:-6])
# print(text_idx)
# #结果是 35

# texts = util.text_file(os.listdir(join('../dataset/Clevr/', 'train' + '_text')))
# texts.sort()
# texts = [join('../dataset/Clevr/', 'train' + '_text', name) for name in texts]
# print(texts)#这是获取了所有的文件地址+文件名
# text = texts[4]
# print(text)
# # text=../dataset/Clevr/train_text\CLEVR_new_000004_1.txt


# texts = np.load('../dataset/Clevr/train_text.npy',allow_pickle=True,encoding='latin1')
#text=texts[-1]
# print(texts) 0维
#'target_color': 'gray', 'type': 'make'}, {'make_new_color': None, 'from': [998, 710, 917], 'make_new_size': 'large', 'to_str': 'make bottom-center small purple circle large', 'target_shape': 'circle', 'target_pos': 'bottom-center', 'to': [19003, 19004, 19005], 'target_size': 'small', 'str': ['make'], 'target_color': 'purple', 'type': 'make'}, {'make_new_color': 'cyan', 'from': [913, 758, 258], 'make_new_size': None, 'to_str': 'make bottom-center brown object cyan', 'target_shape': None, 'target_pos': 'bottom-center', 'to': [19006, 19007, 19008], 'target_size': None, 'str': ['make'], 'target_color': 'brown', 'type': 'make'}, {'make_new_color': None, 'from': [780, 136, 915], 'make_new_size': 'large', 'to_str': 'make bottom-left small brown object large', 'target_shape': None, 'target_pos': 'bottom-left', 'to': [19009, 19010, 19011], 'target_size': 'small', 'str': ['make'], 'target_color': 'brown', 'type': 'make'}]}}
# for text in texts:
#     t = '[CLS] ' + text[0].upper() + text[1:] + ' [SEP]'
#     print(t)

# f = open("../dataset/Clevr/train_text\CLEVR_new_000004_1.txt",encoding = "utf-8")
# #输出读取到的数据
# texts=f.read()
# print(texts)
# texts=texts.split(' ')
# for text in texts:
#     print(text)
#     t = '[CLS] ' + text[0].upper() + text[1:] + ' [SEP]'
#     print(t)