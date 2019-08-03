"""
demo01_fi.py  特征重要性
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pyplot as mp
import numpy as np

# 加载数据
boston = sd.load_boston()
print(boston.data.shape)  # 样本输入
print(boston.target.shape)# 样本输出
print(boston.feature_names) # 特征名
names = boston.feature_names
'''
 |犯罪率|住宅用地比例|商业用地比例|
 |是否靠河|空气质量|房间数|年限|
 |距市中心距离|路网密度|房产税|师生比|
 |黑人比例|低地位人口比例|
'''
# 打乱原始数据集， 划分训练集与测试集
# 当随机种子相同时得到的随机序列也相同
x, y=su.shuffle(boston.data, boston.target, 
	random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:]

# 构建AdaBoost模型 使用训练集训练，测试集测试
tree=st.DecisionTreeRegressor(max_depth=4)
model = se.AdaBoostRegressor(tree, 
	n_estimators=400, random_state=7)
model.fit(train_x, train_y)
ada_fi = model.feature_importances_
print(ada_fi)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 构建DT模型 使用训练集训练，测试集测试
tree=st.DecisionTreeRegressor(max_depth=4)
tree.fit(train_x, train_y)
tree_fi = tree.feature_importances_
print(tree_fi)
pred_test_y = tree.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 绘制特征重要性的柱状图
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('AdaBoost FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = ada_fi.argsort()[::-1]
pos = np.arange(ada_fi.size)
mp.bar(pos, ada_fi[sorted_inds], 0.8,
	facecolor='dodgerblue', label='AdaBoost')
# 设置刻度文本
mp.xticks(pos, names[sorted_inds]) 
mp.legend()

mp.subplot(212)
mp.title('DT FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = tree_fi.argsort()[::-1]
pos = np.arange(tree_fi.size)
mp.bar(pos, tree_fi[sorted_inds], 0.8,
	facecolor='orangered', label='DT')
# 设置刻度文本
mp.xticks(pos, names[sorted_inds]) 
mp.legend()

mp.tight_layout()
mp.show()
