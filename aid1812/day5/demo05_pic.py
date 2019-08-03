"""
demo05_pic.py  图像量化
1. 读取图片灰度图像 把每个像素的亮度值整理进入训练集x。 512*512
2. 把x带入KMeans模型，4个聚类划分，得到聚类中心的值。
3. 对原图片进行处理，修改每个像素的亮度值为最接近的聚类中心值。
4. 图像量化处理结束， 绘制图片。
"""
import numpy as np
import matplotlib.pyplot as mp
import scipy.misc as sm
import scipy.ndimage as sn
import sklearn.cluster as sc

img=sm.imread('../ml_data/lily.jpg',True)
print(img.shape)
x = img.reshape(-1, 1)
print(x.shape)
model = sc.KMeans(n_clusters=4)
model.fit(x)
# 返回类别标签
y = model.labels_
centers = model.cluster_centers_.ravel()
print(y,y[:50],y.shape,end='\n')
print(centers)
# 使用掩码完成量化操作
img2 = centers[y].reshape(img.shape)

mp.figure('Image Quant', facecolor='lightgray')
mp.subplot(121)
mp.xticks([])
mp.yticks([])
mp.imshow(img, cmap='gray')
mp.subplot(122)
mp.xticks([])
mp.yticks([])
mp.imshow(img2, cmap='gray')
mp.tight_layout()
mp.show()


