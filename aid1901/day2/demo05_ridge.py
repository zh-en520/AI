# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_ridge.py  岭回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 把输入变为二维数组，一行一样本，一列一特征
x = x.reshape(-1, 1) # 变为n行1列
model = lm.Ridge(200, fit_intercept=True,
    max_iter=1000)
model.fit(x, y)
# 把样本x带入模型求出预测y
pred_y = model.predict(x)

# 输出模型的评估指标  sm
import sklearn.metrics as sm
print(sm.mean_absolute_error(y, pred_y))
print(sm.mean_squared_error(y, pred_y))
print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y))

# 绘制图像
mp.figure('Ridge Regression', facecolor='lightgray')
mp.title('Ridge Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')
mp.scatter(x, y, c='dodgerblue', s=80,
    marker='o', label='Points')
mp.plot(x, pred_y, c='orangered',
    label='Regression Line')
mp.legend()
mp.show()




