# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_poly.py  多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import sklearn.metrics as sm

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 创建模型
x = x.reshape(-1, 1)
model = pl.make_pipeline(
    sp.PolynomialFeatures(10),
    lm.LinearRegression())
model.fit(x, y)
# 针对所有的x  求预测值pred_y
pred_y = model.predict(x)
print(sm.r2_score(y, pred_y))

# 绘制多项式回归线
px = np.linspace(x.min(), x.max(), 1000)
px = px.reshape(-1, 1)
pred_py = model.predict(px)

# 绘制图像
mp.figure('Poly Regression', facecolor='lightgray')
mp.title('Poly Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')
mp.scatter(x, y, c='dodgerblue', s=80,
    marker='o', label='Points')
mp.plot(px, pred_py, c='orangered', 
    label='PolyFit Line')
mp.legend()
mp.show()




