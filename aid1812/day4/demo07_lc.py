"""
demo07_lc.py  小汽车评级  学习曲线
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp

def f(s):
	return str(s, encoding='utf-8')

data = np.loadtxt('../ml_data/car.txt',
	delimiter=',', dtype='U10',
	converters={0:f,1:f,2:f,3:f,4:f,5:f,6:f})
print(data.shape)
encoders = []   # 存储标签编码器
train_x, train_y = [], []
data = data.T
for row in range(len(data)):
	encoder = sp.LabelEncoder()
	if row < len(data)-1:
		train_x.append(
			encoder.fit_transform(data[row]))
	else:
		train_y = encoder.fit_transform(
				  data[row])
	encoders.append(encoder)
#整理输入输出集
train_x = np.array(train_x).T
train_y = np.array(train_y)

# 训练模型  随机森林分类器
model = se.RandomForestClassifier(
	max_depth=9,
	n_estimators=150, random_state=7)

'''
# 使用学习曲线 得到模型最优训练集大小
train_sizes = np.linspace(0.1, 1, 10)
_, train_scores, test_scores = \
	ms.learning_curve(
		model, train_x, train_y,
		train_sizes=train_sizes, 
		cv=5)
lc_array = np.mean(test_scores, axis=1)
mp.figure('Learning Curve',facecolor='lightgray')
mp.title('Learning Curve', fontsize=16)
mp.xlabel('train size')
mp.ylabel('f1_scores')
mp.grid(linestyle=':')
mp.plot(train_sizes, lc_array)
mp.show()
'''

train_x, test_x, train_y, test_y = \
  ms.train_test_split(train_x, train_y, 
	test_size=0.3, random_state=7)

model.fit(train_x, train_y)

# 自定义测试集，进行模型预测
data = [
['high','med','5more','4','big','low','unacc'],
['high','high','4','4','med','med','acc'],
['low','low','2','4','small','high','good'],
['low','med','3','4','med','high','vgood']]
# 使用相同的标签编码器进行编码 
data = np.array(data).T
test_x = []
for row in range(len(data)):
	# 以前二手的标签编码器
	encoder = encoders[row] 
	if row < len(data)-1:
		test_x.append(
			encoder.transform(data[row]))
test_x = np.array(test_x).T
pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(
			pred_test_y))