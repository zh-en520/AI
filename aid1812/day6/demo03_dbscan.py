"""
demo03_dbscan.py  dbscan
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

x = np.loadtxt('../ml_data/perf.txt',
	delimiter=',')
# 优选参数
epsilons, scores, models = \
	np.linspace(0.3, 1.2, 10), [], []
for epsilon in epsilons:
	model=sc.DBSCAN(eps=epsilon, min_samples=5)
	model.fit(x)
	score = sm.silhouette_score(
		x, model.labels_, sample_size=len(x), 
		metric='euclidean')
	scores.append(score)
	models.append(model)

scores = np.array(scores)
best_index = scores.argmax()
print('best eps:', epsilons[best_index])
print('best model:', models[best_index])
print('best scores:', scores[best_index])

# 获取核心样本、外周样本、孤立样本，并且绘图
besk_model = models[best_index]
pred_y = besk_model.fit_predict(x)
# 获取核心样本的掩码
core_mask = np.zeros(len(x), dtype=bool)
core_mask[besk_model.core_sample_indices_] = True
# 孤立样本的掩码
offset_mask = pred_y == -1
# 外周样本的掩码
periphery_mask = ~(core_mask|offset_mask)

# 画图
mp.figure('DBSCAN Clustering', facecolor='lightgray')
mp.title('DBSCAN Clustering', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x[core_mask][:,0], 
	x[core_mask][:,1], c=pred_y[core_mask],
	s=80, label='Core', cmap='brg')
mp.scatter(x[periphery_mask][:,0], 
	x[periphery_mask][:,1], alpha=0.5,
	c=pred_y[periphery_mask],
	s=70, label='periphery', cmap='brg')
mp.scatter(x[offset_mask][:,0], 
	x[offset_mask][:,1], 
	c='gray',label='offset')

mp.legend()
mp.show()
