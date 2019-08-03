# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_kmeans.py  k均值
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 基于KMeans完成聚类
model = sc.KMeans(n_clusters=4)
model.fit(x) # 完成聚类
pred_y = model.predict(x) # 预测点在哪个聚类中
print(pred_y) # 输出每个样本的聚类标签
# 获取聚类中心
centers = model.cluster_centers_

# 绘制聚类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

# 画图显示样本数据
mp.figure('KMeans', facecolor='lightgray')
mp.title('KMeans', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:,0], x[:,1], s=80,
    c=pred_y, cmap='brg', label='Samples')
mp.scatter(centers[:,0], centers[:,1],
	color='red',
    s=400, marker='+', label='Cluster Center')
mp.legend()
mp.show()




