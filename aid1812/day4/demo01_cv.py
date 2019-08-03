"""
demo01_cv.py  交叉验证
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb

# 加载文件，读取数据
data=np.loadtxt('../ml_data/multiple1.txt',
	unpack=False, delimiter=',')
print(data.shape)
x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 把整个空间进行网格化拆分，通过拆分出来的
# 每个点根据分类模型预测每个点类别名，填充
# 相应的颜色值。 pcolormesh
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n), 
	np.linspace(b, t, n))

# 拆分训练集与测试集
import sklearn.model_selection as ms
train_x, test_x, train_y, test_y = \
	ms.train_test_split(x, y, 
		test_size=0.25, random_state=7)

# 构建NB分类模型
model = nb.GaussianNB()
#进行交叉验证，看一下模型精度，若不错再训练
ac = ms.cross_val_score(model, 
	 	train_x, train_y, cv=5, 
	 	scoring='accuracy')
print(ac.mean())
#进行交叉验证  查准率
pw = ms.cross_val_score(model, 
	 	train_x, train_y, cv=5, 
	 	scoring='precision_weighted')
print(pw.mean())
#进行交叉验证  查准率
rw = ms.cross_val_score(model, 
	 	train_x, train_y, cv=5, 
	 	scoring='recall_weighted')
print(rw.mean())

#进行交叉验证  查准率
f1 = ms.cross_val_score(model, 
	 	train_x, train_y, cv=5, 
	 	scoring='f1_weighted')
print(f1.mean())


model.fit(train_x, train_y)
# 整理结构变为 25万行2列的二维数组
grid_xy = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(grid_xy)
# 整列输出，变维：500*500
grid_z = grid_z.reshape(grid_x.shape)

mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y,
			cmap='brg', s=80)
mp.show()


