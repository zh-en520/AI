"""
demo06_meanshift.py  均值漂移
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/multiple3.txt',
	delimiter=',')
bw = sc.estimate_bandwidth(x, 
	n_samples=len(x), quantile=0.2)
model = sc.MeanShift(bandwidth = bw,
			 bin_seeding = True)
model.fit(x)
pred_y = model.predict(x)
# 获取聚类中心
centers = model.cluster_centers_
print(centers)

# 把整个空间进行网格化拆分，通过拆分出来的
# 每个点根据分类模型预测每个点类别名，填充
# 相应的颜色值。 pcolormesh
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n), 
	np.linspace(b, t, n))
# 整理结构变为 25万行2列的二维数组
grid_xy = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(grid_xy)
# 整列输出，变维：500*500
grid_z = grid_z.reshape(grid_x.shape)

# 画图
mp.figure('KMeans', facecolor='lightgray')
mp.title('KMeans', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:,0], x[:,1], s=60, marker='o', 
	c=pred_y, cmap='brg',
	label='Sample Points')
mp.scatter(centers[:,0], centers[:,1], 
	s=350, marker='+', color='red',
	label='Centers')

mp.legend()
mp.show()


