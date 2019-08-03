"""
demo10_svm_rbf.py  svm  poly核函数
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple2.txt',
	delimiter=',')
x = data[:, :-1]
y = data[:, -1]

# 基于svm 实现分类
model = svm.SVC(kernel='rbf', 
			    C=600, gamma=0.01,
			    probability=True)
model.fit(x, y)

# 新增测试样本点
prob_x = np.array([
	[2, 1.5],
	[8, 9],
	[4.8, 5.2],
	[4, 4],
	[2.5, 7]])
# 预测测试样本的类别，输出测试样本的置信概率
pred_prob_y = model.predict(prob_x)
probs = model.predict_proba(prob_x)
print(probs)

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
mp.figure('SVM Linear', facecolor='lightgray')
mp.title('SVM Linear', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], s=60, marker='o', 
	alpha=0.7, label='Sample Points', 
	c=y, cmap='brg')
mp.scatter(prob_x[:, 0], prob_x[:, 1], 
	s=80, marker='D', color='red',
	label='Prob Points')
mp.legend()
mp.show()








