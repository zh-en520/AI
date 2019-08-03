"""
demo03_dump.py  模型存储
"""
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import pickle
# 读取文件采集数据
x, y = np.loadtxt('../ml_data/single.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)

# 整理训练集 
x = x.reshape(-1, 1) # x变为n行1列
model = lm.LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)

# 评估训练结果误差
print('r2:', sm.r2_score(y, pred_y))
# 模型存储
with open('../ml_data/linear.pkl', 'wb')as f:
	pickle.dump(model, f)

print('Dump Success!')