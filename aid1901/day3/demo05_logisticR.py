# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_logisticR.py 逻辑回归
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([[3, 1], 
              [2, 5], 
              [1, 8], 
              [6, 4], 
              [5, 2], 
              [3, 5], 
              [4, 7], 
              [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 根据找到的规律 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
# 构建逻辑回归模型，并训练模型
import sklearn.linear_model as lm
model = lm.LogisticRegression(
    solver='liblinear', C=1)
model.fit(x, y)
# 把网格坐标矩阵中500*500的所有点做类别预测
test_x = np.column_stack(
    (grid_x.ravel(), grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)
# 绘制样本数据
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification')
mp.xlabel('X')
mp.ylabel('Y')
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')

mp.scatter(x[:, 0], x[:, 1], s=80,
    c=y, cmap='jet', label='Samples')

mp.legend()
mp.show()







