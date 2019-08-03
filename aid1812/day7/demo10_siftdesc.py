"""
demo10_siftdesc.py sift特征矩阵
"""
import cv2 as cv
import matplotlib.pyplot as mp

img = cv.imread('../ml_data/table.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# sift特征点检测器
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
_, desc = sift.compute(gray, keypoints)
print(desc.shape)

mp.matshow(desc.T, cmap='gist_rainbow')
mp.show()