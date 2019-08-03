# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_ac.py  凝聚层次
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 基于凝聚层次完成聚类
model = sc.AgglomerativeClustering(
    n_clusters=4)
pred_y = model.fit_predict(x) # 预测点在哪个聚类中
print(pred_y) # 输出每个样本的聚类标签

# 画图显示样本数据
mp.figure('AgglomerativeClustering', facecolor='lightgray')
mp.title('AgglomerativeClustering', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:,0], x[:,1], s=80,
    c=pred_y, cmap='brg', label='Samples')
mp.legend()
mp.show()




