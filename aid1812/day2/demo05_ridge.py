"""
demo05_ridge.py  岭回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)


# 画图
mp.figure('Ridge Regression', facecolor='lightgray')
mp.title('Ridge Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	label='Sample Points')

# 整理训练集 
x = x.reshape(-1, 1) # x变为n行1列
model = lm.Ridge(50, fit_intercept=True,
	max_iter=1000)
model.fit(x, y)
pred_y = model.predict(x)
print(sm.r2_score(y,pred_y))
mp.plot(x, pred_y, c='orangered', 
	linewidth=2, label='Ridge Regression')


mp.legend()
mp.show()
