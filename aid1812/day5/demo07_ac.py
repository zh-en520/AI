"""
demo07_ac.py  凝聚层次
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/multiple3.txt',
	delimiter=',')
#凝聚层次算法
model = sc.AgglomerativeClustering(
		n_clusters=4)
pred_y = model.fit_predict(x)

# 画图
mp.figure('Agglomerative Clustering', facecolor='lightgray')
mp.title('Agglomerative Clustering', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x[:,0], x[:,1], s=60, marker='o', 
	c=pred_y, cmap='brg',
	label='Sample Points')
mp.legend()
mp.show()


