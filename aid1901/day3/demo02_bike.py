# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_bike.py 分析共享单车需求  
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = []
with open('../ml_data/bike_day.csv', 'r') as f:
    for line in f.readlines():
        data.append(line.split(','))
f.close()
data = np.array(data)
day_headers = data[0, 2:13] 
x = np.array(data[1:, 2:13], dtype='f8')
y = np.array(data[1:, -1], dtype='f8')

# 划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:], 
# 选择随机森林模型 
model = se.RandomForestRegressor(max_depth=10,
    n_estimators=1000, min_samples_split=3)
# 模型训练
model.fit(train_x, train_y)
# 模型测试
pred_test_y = model.predict(test_x)
# 求r2得分
print(sm.r2_score(test_y, pred_test_y))
# 输出模型的特征重要性
day_fi = model.feature_importances_


data = []
with open('../ml_data/bike_hour.csv', 'r') as f:
    for line in f.readlines():
        data.append(line.split(','))
f.close()
data = np.array(data)
hour_headers = data[0, 2:14] 
x = np.array(data[1:, 2:14], dtype='f8')
y = np.array(data[1:, -1], dtype='f8')

# 划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:], 
# 选择随机森林模型 
model = se.RandomForestRegressor(max_depth=10,
    n_estimators=1000, min_samples_split=3)
# 模型训练
model.fit(train_x, train_y)
# 模型测试
pred_test_y = model.predict(test_x)
# 求r2得分
print(sm.r2_score(test_y, pred_test_y))
# 输出模型的特征重要性
hour_fi = model.feature_importances_


mp.figure('Feature Importances', facecolor='lightgray')
mp.subplot(211)
mp.title('Bike_day FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = day_fi.argsort()[::-1]
x = np.arange(day_headers.size)
mp.bar(x, day_fi[sorted_indices], 0.8, 
    color='dodgerblue', label='DTFI')
mp.xticks(x, day_headers[sorted_indices])
mp.legend()

mp.subplot(212)
mp.title('Bike_hour FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = hour_fi.argsort()[::-1]
x = np.arange(hour_headers.size)
mp.bar(x, hour_fi[sorted_indices], 0.8, 
    color='orangered', label='DTFI')
mp.xticks(x, hour_headers[sorted_indices])
mp.legend()

mp.tight_layout()
mp.show()











