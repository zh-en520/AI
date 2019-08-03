"""
demo03_traffic.py  车流量预测
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

class DigitEncoder():
	"""
	模拟LabelEncoder编码器 用于对数字字符串编码
	"""
	def fit_transform(self, y):
		return y.astype(int)

	def transform(self, y):
		return y.astype(int)

	def inverse_transform(self, y):
		return y.astype(str)

def f(s):
	return str(s, encoding='utf-8')

# 整理数据集  
data = np.loadtxt('../ml_data/traffic.txt', 
	delimiter=',', dtype='U20',
	converters={0:f,1:f,2:f,3:f,4:f})
# 编码处理
data = data.T  
encoders, x = [], []
for row in range(len(data)):
	if data[row][0].isdigit():
		encoder = DigitEncoder()
	else:
		encoder = sp.LabelEncoder()
	if row < len(data)-1:
		x.append(
			encoder.fit_transform(data[row]))
	else:
		y = encoder.fit_transform(data[row])
	encoders.append(encoder)
x = np.array(x).T
# 输入集输出集整理完毕  拆分测试/训练集
train_x, test_x, train_y, test_y = \
	ms.train_test_split(x, y, test_size=0.25,
		random_state=5)
model = svm.SVR(kernel='rbf', C=10,
				epsilon=0.2)

# 模型训练
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))


# 找一组没有出现过的数据 测试
data = [['Tuesday', '13:35', 'San Francisco', 'yes']]
data = np.array(data).T  
x = []
for row in range(len(data)):
	encoder = encoders[row]
	x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
print(int(pred_y))
