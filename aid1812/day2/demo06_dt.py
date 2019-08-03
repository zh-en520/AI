"""
demo07_dt.py  决策树 预测波士顿地区房屋价格。
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm

# 加载数据
boston = sd.load_boston()
print(boston.data.shape)  # 样本输入
print(boston.target.shape)# 样本输出
print(boston.feature_names) # 特征名
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

# 构建模型 使用训练集训练，测试集测试
model=st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))