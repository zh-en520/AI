# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_load.py  线性回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import pickle 
# 采集数据
x, y = np.loadtxt('../ml_data/single.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 从文件中加载模型
with open('../ml_data/lr.pkl', 'rb') as f:
    model = pickle.load(f)
# 把样本x带入模型求出预测y
pred_y = model.predict(x.reshape(-1, 1))

# 输出模型的评估指标  sm
import sklearn.metrics as sm
print(sm.mean_absolute_error(y, pred_y))
print(sm.mean_squared_error(y, pred_y))
print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y))

# 绘制图像
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
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




