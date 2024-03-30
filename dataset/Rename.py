# 对文件夹中的图片进行重命名
import os
import os.path


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    # def __init__(self):
    #     self.path =r'F:\lrc\tim-gan-main\tim-gan-ACM\dataset\clevr\images_test_3_edit\test_A'  # 表示需要命名处理的文件夹
    #
    # def rename(self):
    #     filelist = os.listdir(self.path)  # 获取文件路径
    #     total_num = len(filelist)  # 获取文件长度（个数）
    #     i = 0  # 表示文件的命名是从1开始的
    #     for item in filelist:
    #         if item.endswith('.png'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其
    #             # 他格式，后面的转换格式就可以调整为自己需要的格式即可）
    #             src = os.path.join(os.path.abspath(self.path), item)
    #             # dst = os.path.join(os.path.abspath(self.path), '' + str(i).zfill(5) + '.png')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
    #             # dst = os.path.join(os.path.abspath(self.path), '' + str(i) + '.jpg')
    #             # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
    #             str=item.replace('new','source')#一开始这里出错是 要替换的值和替换后的值放反了 前面是old 后面是new的名字
    #             dst=os.path.join(os.path.abspath(self.path),str)
    #             try:
    #                 os.rename(src, dst)
    #                 print('%d converting %s to %s ...' % (i, src, dst))
    #                 i = i + 1
    #             except:
    #                 continue
    #     print('total %d to rename & converted %d jpgs' % (total_num, i))

    def __init__(self):
        self.path =r'F:\lrc\tim-gan-main\tim-gan-ACM\dataset\codraw\final_train_B'  # 表示需要命名处理的文件夹

    def rename(self):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取文件长度（个数）
        i = 0  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.png'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其
                # 他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), '' + str(i).zfill(5) + '.png')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '' + str(i) + '.jpg')
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                str=item.replace('Scene','')#一开始这里出错是 要替换的值和替换后的值放反了 前面是old 后面是new的名字
                dst=os.path.join(os.path.abspath(self.path),str)
                try:
                    os.rename(src, dst)
                    print('%d converting %s to %s ...' % (i, src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()