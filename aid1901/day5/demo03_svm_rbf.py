# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_svm_rbf.py  svm支持向量机
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple2.txt', 
    delimiter=',', dtype='f8')
x = data[:, :-1]
y = data[:, -1]
# 拆分测试集与训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=5)
# 训练svm模型
model = svm.SVC(kernel='rbf', C=600, 
    gamma=0.01, probability=True)
model.fit(train_x, train_y)
# 预测
pred_test_y = model.predict(test_x)
print(sm.classification_report(
        test_y, pred_test_y))

# 自定义一组测试样本 输出样本的置信概率
prob_x = np.array([[2, 1.5],
                   [8, 9],
                   [4.8, 5.2],
                   [4, 4],
                   [2.5, 7],
                   [7.6, 2],
                   [5.4, 5.9]])
pred_prob_y = model.predict(prob_x)
probs = model.predict_proba(prob_x)
print(probs)

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
mp.scatter(prob_x[:,0], prob_x[:,1], s=80,
    c='orange', label='prob Samples')

# 为每一个点添加备注，标注置信概率
for i in range(len(probs)):
    mp.annotate(
        '[{:.2f}, {:.2f}]'.format(
            probs[i, 0], probs[i, 1]),
        xycoords='data', 
        xy=prob_x[i],
        textcoords='offset points',
        xytext=(-50, 30), fontsize=10,
        color='red',
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='angle3'
        )
    )

mp.legend()
mp.show()








