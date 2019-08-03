# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_cv.py  opencv opencv基础
"""
import numpy as np
import cv2 as cv

# 读取图片
img = cv.imread('../ml_data/forest.jpg')
print(type(img), img.shape, img[0, 0, :])
cv.imshow('figure title', img)
# 显示图片某个颜色通道的图像
blue = np.zeros_like(img)
blue[:, :, 0] = img[:, :, 0]
cv.imshow('blue', blue)
green = np.zeros_like(img)
green[:, :, 1] = img[:, :, 1]
cv.imshow('green', green)
red = np.zeros_like(img)
red[:, :, 2] = img[:, :, 2]
cv.imshow('red', red)
# 图像裁剪  ==  三维数组的切片
h, w = img.shape[:2]
l, t = int(w/4), int(h/4)
r, b = int(w/4 * 3), int(h/4 * 3)
cropped = img[t:b, l:r]
cv.imshow('cropped', cropped)
# 图像缩放
scale1 = cv.resize(img, (int(w/4), int(h/2)))
cv.imshow('scale1', scale1)
cv.waitKey()
# 图像保存
cv.imwrite('../ml_data/green.jpg', green)







