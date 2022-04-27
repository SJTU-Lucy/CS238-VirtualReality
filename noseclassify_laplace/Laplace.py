import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

# ori = Image.open(r'results/nst-stroke/04.15.22-09.13.15-1/images/golden-gate-bridge-starry_night_1000_0.jpg')
ori = Image.open(r'images/zzk.jpg')
ori_gray = ori.convert('L')
ori = np.array(ori)
ori_gray = np.array(ori_gray)
weight = ori.shape[0]
height = ori.shape[1]
print(weight, height)
ori_pad = np.pad(ori_gray, ((1, 1), (1, 1)), 'edge')

t1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
img = np.zeros((weight, height))
for i in range(weight - 2):
    for j in range(height - 2):
        img[i, j] = np.sum(ori_pad[i:i + 3, j:j + 3] * t1)
        if img[i, j] < 0:
            img[i, j] = 0

img_sharp = np.zeros((weight, height))
img_sharp = ori_gray - img
cv2.imshow("name", img)
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭
