"""
demo02_rf.py  随机森林实现共享单车需求量预测
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
# 读取数据集

cols = np.arange(2, 16)
data = np.loadtxt('../ml_data/bike_day.csv',
	skiprows=1, unpack=False, 
	delimiter=',', usecols=cols)
day_headers = np.loadtxt('../ml_data/bike_day.csv',
	unpack=False, dtype='U20', delimiter=',')
day_headers = day_headers[0, 2:13]

x = np.array(data[:, 0:11], dtype='f4')
y = np.array(data[:, -1], dtype='f4')
#打乱数据集，划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
print(x.shape, y.shape)
train_size = int(len(x) * 0.9) 
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:], 
# 基于随机森林训练模型
model = se.RandomForestRegressor(
	max_depth=10,
	n_estimators=1000, 
	min_samples_split=2)
model.fit(train_x, train_y)
day_fi = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

data = np.loadtxt('../ml_data/bike_hour.csv',
	unpack=False, dtype='U20',
	delimiter=',')
hour_headers = data[0, 2:14]
x = np.array(data[1:, 2:14], dtype='f4')
y = np.array(data[1:, -1], dtype='f4')
#打乱数据集，划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
print(x.shape, y.shape)
train_size = int(len(x) * 0.9) 
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:], 
# 基于随机森林训练模型
model = se.RandomForestRegressor(
	max_depth=10,
	n_estimators=1000, 
	min_samples_split=2)
model.fit(train_x, train_y)
hour_fi = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 绘制两组数据的特征重要性的柱状图 （子图）
# 绘制特征重要性的柱状图
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('RF Day FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = day_fi.argsort()[::-1]
pos = np.arange(day_fi.size)
mp.bar(pos, day_fi[sorted_inds], 0.8,
	facecolor='dodgerblue', label='Day')
# 设置刻度文本
mp.xticks(pos, day_headers[sorted_inds]) 
mp.legend()

mp.subplot(212)
mp.title('RF Hour FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = hour_fi.argsort()[::-1]
pos = np.arange(hour_fi.size)
mp.bar(pos, hour_fi[sorted_inds], 0.8,
	facecolor='orangered', label='Hour')
# 设置刻度文本
mp.xticks(pos, hour_headers[sorted_inds]) 
mp.legend()

mp.tight_layout()
mp.show()
