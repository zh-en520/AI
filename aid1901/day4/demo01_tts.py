# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_tts.py  训练集测试集划分
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb


data = np.loadtxt('../ml_data/multiple1.txt',
    unpack=False, dtype='f8', delimiter=',')
print(data.shape)

x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 训练集测试集划分  使用训练集训练
# 使用测试集测试，绘制测试集样本图像
import sklearn.model_selection as ms
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=7)

# 训练NB模型 完成分类业务
model = nb.GaussianNB()
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 得到预测输出 可以与真实输出做比较，
# 计算预测的精准度 (预测正确的样本数/总样本数)
ac=(test_y == pred_test_y).sum() / test_y.size
print(ac)  

# 绘制分类边界线
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

# 画图
mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(test_x[:,0], test_x[:,1], s=60, 
    c=test_y, label='Samples', cmap='jet')
mp.legend()
mp.show()

