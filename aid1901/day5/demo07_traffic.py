# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_traffic.py  车流量预测
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

class DigitEncoder():
    # 自定义编码器，针对数字字符串做标签编码

    def fit_transform(self, y):
        return y.astype('i4')

    def transform(self, y):
        return y.astype('i4')

    def inverse_transform(self, y):
        return y.astype('str')

data = []
with open('../ml_data/traffic.txt', 'r')as f:
    for line in f.readlines():
        row = line.split(',')
        row[-1] = row[-1][:-1]
        data.append(row)
f.close()
data = np.array(data)

# 整理输入集与输出集
encoders, x, y = [], [], []
data = data.T
for row in range(len(data)):
    if data[row][0].isdigit():
        encoder = DigitEncoder()
    else:
        encoder = sp.LabelEncoder()

    if row < len(data)-1:
        x.append(
            encoder.fit_transform(data[row]))
    else:
        y = encoder.fit_transform(data[row])

    encoders.append(encoder)
x = np.array(x).T

# 拆分测试集训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=7)

# 交叉验证
model = svm.SVR(kernel='rbf', 
                C=10, epsilon=0.2)

model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print('r2:', sm.r2_score(test_y, pred_test_y))

# 对测试数据进行测试
data = [['Tuesday', '13:30', 'San Francisco', 'yes'],
        ['Thursday', '13:30', 'San Francisco', 'no']]
data = np.array(data).T
test_x = []
for row in range(len(data)):
    encoder = encoders[row]
    test_x.append(encoder.transform(data[row]))
test_x = np.array(test_x).T

pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(pred_test_y))
