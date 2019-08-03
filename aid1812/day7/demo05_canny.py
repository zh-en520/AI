"""
demo05_canny.py  边缘识别
"""
import cv2 as cv
img = cv.imread('../ml_data/chair.jpg',
	cv.IMREAD_GRAYSCALE)
cv.imshow('Img', img)
# 水平方向索贝尔边缘识别
hs = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('HSobel', hs)
# 垂直方向索贝尔边缘识别
vs = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('VSobel', vs)
s = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Sobel', s)

# 拉普拉斯边缘识别
lap = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Laplacian', lap)

# Canny边缘识别
canny = cv.Canny(img, 50, 200)
cv.imshow('canny', canny)

cv.waitKey()



