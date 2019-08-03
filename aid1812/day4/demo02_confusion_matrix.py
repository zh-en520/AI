"""
demo02_confusion_matrix.py  混淆矩阵
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb
import sklearn.metrics as sm
import sklearn.model_selection as ms

# 加载文件，读取数据
data=np.loadtxt('../ml_data/multiple1.txt',
	unpack=False, delimiter=',')
print(data.shape)
x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
	ms.train_test_split(x, y, 
		test_size=0.25, random_state=7)

# 构建NB分类模型
model = nb.GaussianNB()
#模型训练
model.fit(train_x, train_y)
#与测试集进行测试，输出混淆矩阵
pred_test_y = model.predict(test_x)
m = sm.confusion_matrix(test_y, pred_test_y)
print(m)


mp.figure('Confusion Matrix', facecolor='lightgray')
mp.title('Confusion Matrix', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.imshow(m, cmap='gray')
mp.show()


