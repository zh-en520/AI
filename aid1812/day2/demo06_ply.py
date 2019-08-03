"""
demo06_poly.py  多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/single.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)

# 画图
mp.figure('Poly Regression', facecolor='lightgray')
mp.title('Poly Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	label='Sample Points')

# 基于管线实现多项式回归
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.metrics as sm
# 构建模型
model = pl.make_pipeline(
	sp.PolynomialFeatures(10), #多项式特征扩展
	lm.LinearRegression())  #线性回归器
x = x.reshape(-1, 1)
model.fit(x, y)
pred_y = model.predict(x)
# r2得分
print('r2:', sm.r2_score(y, pred_y))
# 绘制曲线
x = np.linspace(x.min(), x.max(), 1000)
x = x.reshape(-1, 1)
test_y = model.predict(x)

mp.plot(x, test_y, c='orangered',
	label='Poly Regression')
mp.legend()
mp.show()
