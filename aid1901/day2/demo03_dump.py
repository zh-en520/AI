# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_dump.py  模型保存
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import pickle

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 把输入变为二维数组，一行一样本，一列一特征
x = x.reshape(-1, 1) # 变为n行1列
print(x.shape)
model = lm.LinearRegression()
model.fit(x, y)

# 保存模型
with open('../ml_data/lr.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Dump Success!')



