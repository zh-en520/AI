"""
demo06_equalizehist.py  直方图均衡化
"""
import cv2 as cv

img = cv.imread('../ml_data/sunrise.jpg')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# 直方图均衡化
equalized_gray = cv.equalizeHist(gray)
cv.imshow('equalized_gray', equalized_gray)
# 对彩色图提高亮度
# YUV：亮度，色度，饱和度
yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
# 单独获取亮度通道，提亮即可
yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0])
color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Color', color)

cv.waitKey()
