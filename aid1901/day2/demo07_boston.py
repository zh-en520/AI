# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_boston.py   波士顿房屋价格预测
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm

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






















