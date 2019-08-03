# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_quant.py  图像量化
1. 读取图片的亮度矩阵
2. 基于KMeans算法完成聚类，获取4个聚类中心的值
3. 修改图片，把每个像素亮度值都改为相应类别的均值
4. 绘制
"""
import numpy as np
import scipy.misc as sm
import scipy.ndimage as sn
import sklearn.cluster as sc
import matplotlib.pyplot as mp

img = sm.imread('../ml_data/lily.jpg', True)
# 基于KMeans完成聚类
model = sc.KMeans(n_clusters=4)
x = img.reshape(-1, 1)
print(x.shape)
model.fit(x)
# 同model.predict(x) 返回每个样本的类别标签
y = model.labels_
print(y)
centers = model.cluster_centers_
img2 = centers[y].reshape(img.shape)
print(centers)

# 绘图
mp.subplot(121)
mp.imshow(img, cmap='gray')
mp.axis('off')  # 关闭坐标轴
mp.subplot(122)
mp.imshow(img2, cmap='gray')
mp.axis('off')  # 关闭坐标轴
mp.tight_layout()
mp.show()








