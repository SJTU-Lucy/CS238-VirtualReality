'''
f = open("list_attr_celeba.txt")
newBigTxt = "bignose.txt"
newBigf = open(newBigTxt, "a+")
newPointyTxt = "pointynose.txt"
newPointyf = open(newPointyTxt, "a+")

# 跳过第一行202599
line = f.readline()
# 跳过第二行属性名称
line = f.readline()
# 第三行开始操作
line = f.readline()
while line:
    array = line.split()
    # 大鼻子BigNose
    if array[8] != "-1":
        new_context = array[0] + '\n'
        newBigf.write(new_context)
    # 尖鼻子PointyNose
    elif array[28] != "-1":
        new_context = array[0] + '\n'
        newPointyf.write(new_context)
    line = f.readline()

f.close()
newBigf.close()
newPointyf.close()
'''
'''
import os
import shutil

bigf = open("bignose.txt")
pointyf = open("pointynose.txt")

bigLine = bigf.readline()
pointyLine = pointyf.readline()

list = os.listdir("img_align_celeba")
bigGo = True
pointyGo = True

for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != ".jpg":
        continue
    bigArray = bigLine.split()
    if len(bigArray) < 1:
        bigGo = False
    pointyArray = pointyLine.split()
    if len(pointyArray) < 1:
        pointyGo = False

    if bigGo and imgName == bigArray[0]:
        oldname = "img_align_celeba/" + imgName
        newname = "data/BigNose/" + imgName
        shutil.move(oldname, newname)
        bigLine = bigf.readline()
    elif pointyGo and imgName == pointyArray[0]:
        oldname = "img_align_celeba/" + imgName
        newname = "data/PointyNose/" + imgName
        shutil.move(oldname, newname)
        pointyLine = pointyf.readline()

bigf.close()
pointyf.close()
'''

import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取 flower_photos 文件夹下除 .txt 文件以外所有文件夹名（即5种花的类名）
file_path = 'data'
classes = ["BigNose", "PointyNose"]

for cla in classes:
    mkfile('dataset/train/' + cla)

for cla in classes:
    mkfile('dataset/val/' + cla)

# 划分比例，训练集 : 验证集 = 9 : 1
split_rate = 0.1

# 遍历5种花的全部图像并按比例分成训练集和验证集
for cla in classes:
    cla_path = file_path + '/' + cla + '/'  # 某一类别花的子目录
    images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'dataset/val/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'dataset/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")