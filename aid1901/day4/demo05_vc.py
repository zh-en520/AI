# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_vc.py  验证曲线选取超参数
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp

data = []
with open('../ml_data/car.txt', 'r') as f:
    for line in f.readlines():
        sample = line.split(',')
        sample[-1] = sample[-1][:-1]
        data.append(sample)
f.close()
data = np.array(data)
# 整理好每一列的标签编码器encoders
# 整理好训练输入集与输出集
data = data.T
encoders = []
train_x, train_y = [], [] 
for row in range(len(data)):
    encoder = sp.LabelEncoder()
    if row < len(data)-1:  # 不是最后一列
        train_x.append(
            encoder.fit_transform(data[row]))
    else:  # 是最后一列  作为输出集
        train_y = encoder.fit_transform(data[row])

    encoders.append(encoder)
train_x = np.array(train_x).T

# 训练随机森林分类器
model = se.RandomForestClassifier(max_depth=6,
    n_estimators=150, random_state=7)

# # 获取n_estimators的验证曲线
# train_scores, test_scores=ms.validation_curve(
#     model, train_x, train_y, 
#     'n_estimators',
#     np.arange(50, 550, 50), cv=5)
# print(np.mean(test_scores, axis=1))

train_scores, test_scores=ms.validation_curve(
    model, train_x, train_y, 
    'max_depth',
    np.arange(1, 7), cv=5)
print(np.mean(test_scores, axis=1))

# 训练之前先交叉验证
cv=ms.cross_val_score(model, train_x, train_y,
    cv=4, scoring='f1_weighted')
print(cv.mean())
model.fit(train_x, train_y)

# 自定义测试数据  预测小汽车等级
# 保证每个特征使用的标签编码器与训练时使用
# 的标签编码器匹配。
data = [
['high','med','5more','4','big','low','unacc'],
['high','high','4','4','med','med','acc'],
['low','low','2','4','small','high','good'],
['low','med','3','4','med','high','vgood']]
data = np.array(data).T
test_x, test_y = [], []
for row in range(len(data)):
    encoder = encoders[row]# 每列对应的编码器
    if row < len(data)-1:
        test_x.append(
            encoder.transform(data[row]))
    else:
        test_y = encoder.transform(data[row])
test_x = np.array(test_x).T

pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(test_y))
print(encoders[-1].inverse_transform(pred_test_y))

# 画图显示验证曲线
mp.figure('Validation Curve', facecolor='lightgray')
mp.title('Max_depth',fontsize=16)
mp.xlabel('Max_depth')
mp.ylabel('f1 score')
mp.grid(linestyle=':')
mp.plot(np.arange(1, 7), 
	np.mean(test_scores, axis=1), 'o-',
	label='Max_depth VC')
mp.legend()
mp.show()
















