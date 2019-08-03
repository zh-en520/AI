"""
demo08_star.py 特征点检测器
"""
import cv2 as cv
img = cv.imread('../ml_data/table.jpg')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# star特征点检测器
star = cv.xfeatures2d.StarDetector_create()
keypoints = star.detect(gray)
print(keypoints)
mixture = img.copy()
# 把特征点都输出在mixture图中
cv.drawKeypoints(img, keypoints, 
  mixture, flags= \
  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Mixture', mixture)

cv.waitKey()
