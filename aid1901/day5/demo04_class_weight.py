# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_class_weight.py  样本类别均衡化
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/imbalance.txt', 
    delimiter=',', dtype='f8')
x = data[:, :-1]
y = data[:, -1]
# 拆分测试集与训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=5)
# 训练svm模型
model = svm.SVC(kernel='rbf',
    class_weight='balanced')
model.fit(train_x, train_y)
# 预测
pred_test_y = model.predict(test_x)
print(sm.classification_report(
        test_y, pred_test_y))

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

# 画图显示样本数据
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(test_x[:,0], test_x[:,1], s=60, 
    c=test_y, label='Samples', cmap='jet')
mp.legend()
mp.show()








