# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_eqhist.py  直方图均衡化
"""
import cv2 as cv

original = cv.imread('../ml_data/sunrise.jpg')
cv.imshow('Original', original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
equalized_gray = cv.equalizeHist(gray)
cv.imshow('Equalized Gray', equalized_gray)
# YUV：亮度，色度，饱和度
yuv = cv.cvtColor(original, cv.COLOR_BGR2YUV)
# 单独取出亮度通道完成直方图均衡化
yuv[..., 0] = cv.equalizeHist(yuv[..., 0])
# yuv[..., 1] = cv.equalizeHist(yuv[..., 1])
# yuv[..., 2] = cv.equalizeHist(yuv[..., 2])
equalized_color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Equalized Color', equalized_color)
cv.waitKey()