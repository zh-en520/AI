"""
demo02_event.py  事件预测
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
data = np.loadtxt('../ml_data/events.txt', 
	delimiter=',', dtype='U10',
	converters={0:f,1:f,2:f,3:f,4:f,5:f})
data = np.delete(data, 1, axis=1)
# 编码处理
data = data.T  #(5, 4060)
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
model = svm.SVC(kernel='rbf', 
				class_weight='balanced')
# 交叉验证
ac = ms.cross_val_score(model, train_x, 
	train_y, cv=3, scoring='accuracy').mean()
print(ac)

# 模型训练
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
cr=sm.confusion_matrix(test_y, pred_test_y)
print(cr)
print((pred_test_y==test_y).sum() / test_y.size)

# 找一组没有出现过的数据 测试
data = [['Tuesday', '13:30:00', '21', '23']]
data = np.array(data).T  
x = []
for row in range(len(data)):
	encoder = encoders[row]
	x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
# print(encoders[-1].inverse_transform(pred_y))
a = encoders[-1].inverse_transform(pred_y)
# b = np.ones_like(a.shape)
# print(np.divide(a,b))
print(int(pred_y))
print(a)