# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_dbscan.py  dbscan聚类
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 选择最优半径
epsilons = np.linspace(0.3, 1.0, 8)
models, scores = [], []
for epsilon in epsilons:
    # 针对每个半径构建DBSCAN模型
    model=sc.DBSCAN(eps=epsilon, min_samples=5)
    model.fit(x)
    print(model.labels_)
    score = sm.silhouette_score(x, 
        model.labels_, sample_size=len(x), 
        metric='euclidean')
    models.append(model)
    scores.append(score)
index = np.array(scores).argmax()
best_model = models[index]
best_eps = epsilons[index]
best_score = scores[index]
print(best_eps, best_model, best_score)

# DBSCAN算法的副产品  获取所有核心样本的下标
core_indices=best_model.core_samples_indices_
core_mask = np.zeros(len(x), dtype='bool')
core_mask[core_indices] = True
# 孤立样本的掩码
offset_mask = best_model.labels_==-1
# 外周样本的掩码
p_mask = ~(core_mask | offset_mask)


# 画图显示样本数据
mp.figure('DBSCAN', facecolor='lightgray')
mp.title('DBSCAN', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:,0], x[:,1], s=80,
    c=best_model.labels_, cmap='brg', label='Samples')
mp.legend()
mp.show()




