"""
绘制损失函数的三维图像
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

n = 1000
# 构建网格点坐标矩阵
w0, w1 = np.meshgrid(np.linspace(-10,10,n),
				   np.linspace(-10,10,n))
# 根据每个坐标点的x与y计算高度值z
# z = (1-x/2+x**5+y**3) * np.exp(-x**2-y**2)

loss = np.zeros_like(w0)
x = [0.5, 0.6, 0.8, 1.1, 1.4]
y = [5.0, 5.5, 6.0, 6.8, 7.0]
for a, b in zip(x, y):
	loss += 1/2 * (w0 + w1*a - b)**2

mp.figure('3D Surface', facecolor='lightgray')
mp.tick_params(labelsize=10)
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(w0, w1, loss, cmap='jet',
	rstride=30, cstride=30)
mp.tight_layout()
mp.show()