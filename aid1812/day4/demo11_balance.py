"""
demo11_balance.py  样本类别均衡化
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/imbalance.txt',
	delimiter=',')
x = data[:, :-1]
y = data[:, -1]

# 基于svm 实现分类
model = svm.SVC(kernel='rbf',gamma=0.02,
				class_weight='balanced')
model.fit(x, y)
pred_y = model.predict(x)
print(sm.classification_report(y, pred_y))

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

# 绘图
mp.figure('SVM Class Balanced', facecolor='lightgray')
mp.title('SVM Class Balanced', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], s=60, marker='o', 
	alpha=0.7, label='Sample Points', 
	c=y, cmap='brg')
mp.legend()
mp.show()








