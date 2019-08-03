"""
demo02_lr.py  线性回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/single.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)

# 整理训练集 
x = x.reshape(-1, 1) # x变为n行1列
model = lm.LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)

# 评估训练结果误差
import sklearn.metrics as sm
print('l1:',sm.mean_absolute_error(y, pred_y))
print('l2:',sm.mean_squared_error(y, pred_y))
print('ml1:',sm.median_absolute_error(y, pred_y))
print('r2:', sm.r2_score(y, pred_y))
# 画图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	alpha=0.7, label='Sample Points')
mp.plot(x, pred_y, c='red', linewidth=2,
	label='Regression Line')
mp.legend()
mp.show()
