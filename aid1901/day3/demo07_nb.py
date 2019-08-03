# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_nb.py  朴素贝叶斯分类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb


data = np.loadtxt('../ml_data/multiple1.txt',
    unpack=False, dtype='f8', delimiter=',')
print(data.shape)

x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 训练NB模型 完成分类业务
model = nb.GaussianNB()
model.fit(x, y)
# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
test_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)

# 画图
mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:,0], x[:,1], s=60, c=y, 
	label='Samples', cmap='jet')
mp.legend()
mp.show()

