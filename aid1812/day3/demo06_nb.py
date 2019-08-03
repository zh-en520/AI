"""
demo06_nb.py  朴素贝叶斯
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
# 构建逻辑分类模型
model = nb.GaussianNB()
model.fit(x, y)
# 整理结构变为 25万行2列的二维数组
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
print(pred_test_y)
# 整列输出，变维：500*500
grid_z = pred_test_y.reshape(grid_x.shape)
print(grid_z)

mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()


