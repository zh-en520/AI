"""
demo04_cv.py   opencv基础
"""
import numpy as np
import cv2 as cv

# 读取图片并显示
img = cv.imread('../ml_data/forest.jpg')
print(img.shape)
cv.imshow('Forest.jpg', img)
# 显示图片某个颜色通道的图像
blue = np.zeros_like(img)
blue[:,:,0] = img[:,:,0] #保留了蓝色通道的亮度
cv.imshow('Blue', blue)

green = np.zeros_like(img)
green[:,:,1] = img[:,:,1] #保留了绿色通道的亮度
cv.imshow('Green', green)

red = np.zeros_like(img)
red[:,:,2] = img[:,:,2] #保留了红色通道的亮度
cv.imshow('Red', red)

# 图像裁剪
h, w = img.shape[:2]
l, t = int(w/4), int(h/4)
r, b = int(w*3/4), int(h*3/4)
cropped = img[t:b, l:r]
cv.imshow('Cropped', cropped)

# 图像缩放
s1 = cv.resize(img, (int(w/4), int(h/4)),
	interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled1', s1)

s2 = cv.resize(s1, None, fx=4, fy=4,
	interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled2', s2)

cv.waitKey()  # 阻塞方法

# 图像保存
cv.imwrite('../ml_data/red.jpg', red)
cv.imwrite('../ml_data/s2.jpg', s2)
