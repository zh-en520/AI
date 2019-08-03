"""
demo07_corner.py 角点检测
"""
import cv2 as cv

img = cv.imread('../ml_data/box.png')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# 角点检测
corners = cv.cornerHarris(gray, 5, 7, 0.04)
print(corners[corners>0.1])
mixture = img.copy()
mixture[corners>corners.max()*0.01] = \
	(0,0,255)
cv.imshow('Mixture', mixture)

cv.waitKey()
