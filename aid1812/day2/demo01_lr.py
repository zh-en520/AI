"""
demo09_lr.py 线性回归
"""
import numpy as np
import matplotlib.pyplot as mp
import mpl_toolkits.mplot3d as axes3d

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 梯度下降计算回归线
times = 1000  # 下降次数
lrate = 0.03  # 每次下降的学习率 learn_rate
w0, w1 = [1], [1]
epoches = []
losses = []
for i in range(1, times+1):
	epoches.append(i)
	loss = (((w0[-1] + w1[-1]*train_x) \
			-train_y)**2).sum() / 2
	losses.append(loss)
	print('{:4}>w0={:.7f},w1={:.7f},\
		loss={:.7f}'.format(
			i, w0[-1], w1[-1], loss))
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

# 绘制w0，w1，loss的变化曲线
mp.figure('Training Progress', facecolor='lightgray')
mp.subplot(311)
mp.title('Training Progress', fontsize=14)
mp.ylabel('w0', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, w0[:-1], c='dodgerblue',
	label='w0')
mp.legend()

mp.subplot(312)
mp.ylabel('w1', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, w1[:-1], c='dodgerblue',
	label='w1')
mp.legend()

mp.subplot(313)
mp.ylabel('loss', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, losses, c='dodgerblue',
	label='loss')
mp.legend()
mp.tight_layout()

# 在三维曲面图中绘制梯度下降的路线
grid_w0, grid_w1 = np.meshgrid(
	np.linspace(0, 9, 500),
	np.linspace(0, 3.5, 500))
grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
	grid_loss += \
	 ((grid_w0 + x*grid_w1 - y)**2) / 2

mp.figure('Loss Function')
ax = mp.gca(projection='3d')
ax.set_xlabel('w0', fontsize=12)
ax.set_ylabel('w1', fontsize=12)
ax.set_zlabel('loss', fontsize=12)
ax.plot_surface(grid_w0, grid_w1, 
	grid_loss, rstride=10, cstride=10,
	cmap='jet')
ax.plot(w0[:-1], w1[:-1], losses, 'o-',
	c='red', zorder=3)
mp.tight_layout()

# 以等高线图的方式绘制梯度下降的过程
mp.figure('Contour', facecolor='lightgray')
mp.title('Contour', fontsize=14)
mp.xlabel('w0', fontsize=12)
mp.ylabel('w1', fontsize=12)
mp.grid(linestyle=':')
mp.contourf(grid_w0, grid_w1, grid_loss,10,
	cmap='jet')
mp.plot(w0, w1, 'o-', c='red', label='BGD')
mp.legend()

mp.show()



