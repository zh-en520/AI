"""
demo09_lr.py 线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 梯度下降计算回归线
times = 1000  # 下降次数
lrate = 0.01  # 每次下降的学习率 learn_rate
w0, w1 = [1], [1]
for i in range(1, times+1):
	loss = (((w0[-1] + w1[-1]*train_x) \
			-train_y)**2).sum() / 2
	print(w0[-1], ' ', w1[-1], ' ', loss)
	# 根据两个方向的偏导数公式 
	# 计算w0和w1移动的距离 得到新的w0和w1
	d0 = ((w0[-1] + w1[-1]*train_x) \
			- train_y).sum()
	d1 = (((w0[-1]+w1[-1]*train_x) \
			-train_y)*train_x).sum()
	w0.append(w0[-1] - lrate*d0)
	w1.append(w1[-1] - lrate*d1)

k, b = w1[-1], w0[-1]
pred_y = k * train_x + b

mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, c='dodgerblue',
	marker='D', s=60, label='Samples')
mp.plot(train_x, pred_y, c='orangered',
	label='Regression Line')
mp.legend()
mp.show()



