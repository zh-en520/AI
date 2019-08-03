"""
demo03_sc.py  简单人工分类
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([
	[3, 1],
	[2, 5],
	[1, 8],
	[6, 4],
	[5, 2],
	[3, 5],
	[4, 7],
	[4,-1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 把整个空间进行网格化拆分，通过拆分出来的
# 每个点根据分类模型预测每个点类别名，填充
# 相应的颜色值。 pcolormesh
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n), 
	np.linspace(b, t, n))
grid_z = np.piecewise(grid_x, 
		[grid_x>grid_y, grid_x<=grid_y],
		[0, 1])

mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()




