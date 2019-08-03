"""
demo01_gridsearch.py  网格搜索
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
model = svm.SVC(probability=True)
# 基于网格搜索获取最优模型
params = [
	{'kernel':['linear'],'C':[1,10,100,1000]},
	{'kernel':['poly'],'C':[1],'degree':[2,3]},
	{'kernel':['rbf'],'C':[1,10,100,1000], 
	 'gamma':[1,0.1, 0.01, 0.001]}]
model = ms.GridSearchCV(model, params, cv=5)	 
model.fit(x, y)
# 网格搜索训练后的副产品
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_)
# 输出网格搜索每组超参数的cv数据
# for p, s in zip(model.cv_results_['params'],
# 	model.cv_results_['mean_test_score']):
	# print(p, s)
print(model.cv_results_['params'][0])
print(model.cv_results_['mean_test_score'][0])
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
probs1 = model.predict(prob_x)
print(probs)
print(probs1)



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








