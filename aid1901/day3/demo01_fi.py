# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_fi.py  特征重要性指标
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pyplot as mp
import numpy as np

# 加载波士顿地区房屋价格
boston = sd.load_boston()
# ['犯罪率' '住宅地比例' '商业用地比例' 
#  '是否靠河' '空气质量' '房间数' '年限' 
#  '距市中心的距离' '路网密度' '房产税' 
#  '师生比' '黑人比例' '低地位人口比例']

print(boston.feature_names) # 特征名
print(boston.data.shape) # 数据的输入
print(boston.target.shape)  # 数据的输出

# 划分测试集与训练集    80%做训练
# random_state 若打乱时使用的随机种子相同，
# 则得到的结果相同。
x, y = su.shuffle(boston.data, 
    boston.target, random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]
print(train_x.shape)
print(test_x.shape)

# 基于决策树建模->训练模型->测试模型
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
#获取特征重要性指标
dt_fi = model.feature_importances_
print('dt_fi:',dt_fi)

# 基于正向激励模型预测房屋价格
model = se.AdaBoostRegressor(model, 
    n_estimators=400, random_state=7)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
#获取特征重要性指标
ad_fi = model.feature_importances_
print('ad_fi:',ad_fi)

# 绘图
mp.figure('Feature Importances', facecolor='lightgray')
mp.subplot(211)
mp.title('Decision Tree FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
names = boston.feature_names
sorted_indices = dt_fi.argsort()[::-1]
x = np.arange(names.size)
mp.bar(x, dt_fi[sorted_indices], 0.8, 
    color='dodgerblue', label='DTFI')
mp.xticks(x, names[sorted_indices])
mp.legend()

mp.subplot(212)
mp.title('AdaBoost FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = ad_fi.argsort()[::-1]
mp.bar(x, ad_fi[sorted_indices], 0.8, 
    color='orangered', label='ADFI')
mp.xticks(x, names[sorted_indices])
mp.legend()

mp.show()



























