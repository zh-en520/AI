# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_lr.py  梯度下降实现线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
# 实现梯度下降过程
times = 1000
lrate = 0.01
w0, w1 = [1], [1] # 记录每次梯度下降的参数
losses = [] # 保存每次迭代损失函数值
epoches = []  # 保存每次迭代过程的索引
for i in range(1, times+1):
    # 输出每次下降时：w0 w1 loss值的变化
    epoches.append(i)
    loss = ((w0[-1] + w1[-1]*train_x \
             - train_y)**2).sum() / 2
    losses.append(loss)
    print('{:4}> w0={:.6f}, w1={:.6f}, ' \
          'loss={:.6f}'.format(
          epoches[-1], w0[-1], w1[-1],
          losses[-1]))

    # 每次梯度下降过程需要求出w0与w1的修正值
    # 求修正值需要推导loss函数在w0及w1方向的偏导数
    d0=(w0[-1] + w1[-1]*train_x - train_y).sum()
    d1=((w0[-1] + w1[-1]*train_x - train_y)\
        * train_x).sum()
    w0.append(w0[-1] - lrate*d0)
    w1.append(w1[-1] - lrate*d1)

print(w0[-1], w1[-1])
pred_y = w0[-1] + w1[-1]*train_x

# 绘制样本点
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression')
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, s=60, 
    marker='o', c='orangered', label='Samples')
# 绘制回归线
mp.plot(train_x, pred_y, color='dodgerblue',
    label='Regression Line')
mp.legend()

#绘制随着每次梯度下降，w0，w1，loss的变化曲线。
mp.figure('BGD Params', facecolor='lightgray')
mp.title('BGD Params')
mp.tick_params(labelsize=8)
mp.subplot(311)
mp.grid(linestyle=':')
mp.plot(epoches, w0[:-1], color='dodgerblue',
    label='w0')
mp.legend()
mp.subplot(312)
mp.grid(linestyle=':')
mp.plot(epoches, w1[:-1], color='dodgerblue',
    label='w1')
mp.legend()
mp.subplot(313)
mp.grid(linestyle=':')
mp.plot(epoches, losses, color='dodgerblue',
    label='loss')
mp.legend()
mp.tight_layout()

# 基于三维曲面绘制梯度下降过程中的每一个点。
import mpl_toolkits.mplot3d as axes3d
# 整理网格点坐标矩阵，计算每个点的loss绘制曲面
grid_w0, grid_w1 = np.meshgrid(
    np.linspace(0, 9, 500),
    np.linspace(0, 3.5, 500))
grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
    grid_loss += \
        ((grid_w0 + grid_w1*x - y)**2)/2
mp.figure('Loss Function',facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(grid_w0, grid_w1, 
    grid_loss, cmap='jet')
ax3d.plot(w0[:-1], w1[:-1], losses, 
    'o-', c='orangered', label='BGD')
mp.tight_layout()

# 以等高线的方式绘制梯度下降的过程。
mp.figure('BGD Contour', facecolor='lightgray')
mp.title('BGD Contour')
mp.xlabel('w0')
mp.ylabel('w1')
mp.grid(linestyle=':')
cntr=mp.contour(grid_w0, grid_w1, grid_loss,
    c='black', linewidths=0.5)
mp.clabel(cntr, fmt='%.2f', 
    inline_spacing=0.2, fontsize=8)
mp.contourf(grid_w0, grid_w1, grid_loss,
    cmap='jet')
mp.plot(w0[:-1], w1[:-1], 'o-',
    c='orangered', label='BGD')

mp.show()











