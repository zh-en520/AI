# 机器学习DAY05

#### 网格搜索

网格搜索（GridSearch）用于选取模型的最优超参数。获取最优超参数的方式可以绘制验证曲线，但是验证曲线只能每次获取一个最优超参数。如果多个超参数有很多排列组合的话，就可以使用网格搜索寻求最优超参数的组合。

网格搜索针对超参数组合列表中的每一个组合，实例化给定的模型，做cv次交叉验证，将平均f1得分最高的超参数组合作为最佳的选择，返回模型对象。

```python
import sklearn.model_selection as ms
# 经过网格搜索后得到最优超参数组合返回模型
model = ms.GridSearchCV( 
    model, 
    超参数的组合列表, 
	cv=5)
model.fit(....)
# 获取网格搜索时的副产品
model.best_params_		# 最优超参数
model.best_score_		# 最后模型得分
model.best_estimator_  	# 最优模型对象
# 获取所有参数组合
model.cv_results_['params']
# 获取所有参数组合cv后的结果
model.cv_results_['mean_test_score']
```

案例：修改rbf案例，基于网格搜索得到最优模型。

```python

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
for p, s in zip(model.cv_results_['params'],
	model.cv_results_['mean_test_score']):
	print(p, s)
```

#### 案例：事件预测

event.txt，预测某个时间段是否会出现特殊事件。

```python
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
data = np.loadtxt('../ml_data/event.txt', 
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
# 交叉验证_proba
ac = ms.cross_val_score(model, train_x, 
	train_y, cv=3, scoring='accuracy').mean()
print(ac)

# 模型训练
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
cr=sm.confusion_matrix(test_y, pred_test_y)
print(cr)

# 找一组没有出现过的数据 测试
data = [['Tuesday', '13:30:00', '21', '23']]
data = np.array(data).T  
x = []
for row in range(len(data)):
	encoder = encoders[row]
	x.append(encoder.transform(data[row]))
x = np.array(x).T
pred_y = model.predict(x)
print(encoders[-1].inverse_transform(pred_y))
```

#### 案例：交通流量的预测（回归）

```python
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
```

回归问题：线性回归、岭回归、多项式回归、决策树、正向激励决策树、随机森林决策树、SVR

分类问题：简单分类、逻辑分类、决策树分类、朴素贝叶斯、SVM

### 聚类

分类（class）与聚类（cluster）不同，分类属于有监督学习模型，聚类属于无监督学习模型。聚类讲究实用一些算法把样本划分成n个群落。一般情况下，聚类算法需要计算欧氏距离。

欧氏距离：
$$
P(x_1)-Q(x_2):|x_1-x_2|=\sqrt{(x_1-x_2)^2} \\
P(x_1,y_1)-Q(x_2,y_2):\sqrt{(x_1-x_2)^2+(y_1-y_2)^2} \\
P(x_1,y_1,z_1)-Q(x_2,y_2,z_2):\sqrt{(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2} \\
$$


#### KMeans算法（K均值）

第一步：随机选择K个样本作为K个聚类的中心，计算每个样本到各个聚类中心的欧氏距离，将该样本分配到与之距离最近的聚类中心所在类别中。

第二步：根据第一步得到的聚类划分情况，分别计算每个聚类的中心中心作为新的聚类中心，重复第一步，直到计算所得几何中心与聚类中心吻合或接近重合为止。

**注意：**

1. KMean算法的聚类数必须事先已知。可以借助某些评估指标，优选最好的聚类数。
2. 聚类中心的初始选择会影响到最终聚类划分的结果。所以尽量选择距离较远的样本。

```python
import sklearn.cluster as sc
# 聚类模型对象  n_clusters：聚类数
model = sc.KMeans(n_clusters=4)
model.fit(x)
# 返回的即是每个样本的聚类标签
pred_y = model.predict(x)
# 聚类模型训练后的副产品
centers = model.cluster_centers_
```

案例：加载multiple3.txt, 完成聚类。

```python
'''
1. 读取文件数据，基于KMeans训练模型
2. 预测所有的样本，得到每个样本的类别并画图
3. 绘制聚类效果的背景
4. 获取聚类中心，绘制聚类中心+
'''
"""
demo04_kmeans.py  kmeans聚类
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = np.loadtxt('../ml_data/multiple3.txt',
	delimiter=',')
model = sc.KMeans(n_clusters=4)
model.fit(x)
pred_y = model.predict(x)

# 获取聚类中心
centers = model.cluster_centers_
print(centers)

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

# 画图
mp.figure('KMeans', facecolor='lightgray')
mp.title('KMeans', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:,0], x[:,1], s=60, marker='o', 
	c=pred_y, cmap='brg',
	label='Sample Points')
mp.scatter(centers[:,0], centers[:,1], 
	s=350, marker='+', color='red',
	label='Centers')

mp.legend()
mp.show()
```

##### 图像量化

KMeans聚类算法可以应用于图像量化领域。通过KMeans算法可以把一张图像所包含的颜色值进行聚类划分，求每一类别的平均值后再重新生成新的图像。可以达到图像降维的目的。这个过程称为图像量化。

图像量化可以更好的保留图像的轮廓，降低机器识别图像轮廓的难度。

案例：lily.jpg

```python
"""
demo05_pic.py  图像量化
1. 读取图片灰度图像 把每个像素的亮度值整理进入训练集x。 512*512
2. 把x带入KMeans模型，4个聚类划分，得到聚类中心的值。
3. 对原图片进行处理，修改每个像素的亮度值为最接近的聚类中心值。
4. 图像量化处理结束， 绘制图片。
"""
import numpy as np
import matplotlib.pyplot as mp
import scipy.misc as sm
import scipy.ndimage as sn
import sklearn.cluster as sc

img=sm.imread('../ml_data/lily.jpg',True)
x = img.reshape(-1, 1)
model = sc.KMeans(n_clusters=4)
model.fit(x)
# 返回类别标签
y = model.labels_
centers = model.cluster_centers_.ravel()
print(y, y.shape)
print(centers)
# 使用掩码完成量化操作
img2 = centers[y].reshape(img.shape)

mp.figure('Image Quant', facecolor='lightgray')
mp.subplot(121)
mp.xticks([])
mp.yticks([])
mp.imshow(img, cmap='gray')
mp.subplot(122)
mp.xticks([])
mp.yticks([])
mp.imshow(img2, cmap='gray')
mp.tight_layout()
mp.show()
```

#### 均值漂移算法

首先假定样本空间中每个聚类均服从某种已知的概率分布规则，然后用不同的概率密度函数拟合样本中的统计直方图，不断移动密度函数的中心（均值）位置，直到获得最佳拟合效果为止。

这些概率密度函数的峰值点就是聚类的中心。再根据每个样本距离各个中心的距离，选择最近的聚类中心所属类别作为该样本的类别。

均值漂移算法的特点：

1. 聚类中心不必事先已知，算法会自动识别出统计直方图的中心数量。
2. 聚类中心不依据最初的假定初始位置，聚类划分的结果相对稳定。
3. 样本空间应该服从某种概率分布规则，否则算法性能将大大折扣。

```python
# 量化带宽对象
bw = sc.estimate_bandwidth(
	x, n_samples=len(x),
    quantile=0.1)
# bandwidth: 量化带宽对象
# bin_seeding:若为True，则根据量化带宽拟合
# 统计直方图，不直接使用样本统计直方图数据。
model = sc.MeanShift(
    bandwidth=bw, bin_seeding=True)
```

案例：multiple3.txt

```python
bw = sc.estimate_bandwidth(x, 
	n_samples=len(x), quantile=0.2)
model = sc.MeanShift(bandwidth = bw,
			 bin_seeding = True)
model.fit(x)
pred_y = model.predict(x)
```

#### 凝聚层次算法

首先假定每个样本都是一个独立的聚类，如果统计出来的聚类数大于期望的聚类数，则从每个样本触发，寻找离自己最近的另一个样本，与之聚集，形成更大的聚类，同时令总聚类数减少，不断重复以上过程，直到统计出来的聚类数达到期望为止。

凝聚层次的特点：

1. 聚类数必须事先已知。借助某些评估指标，优选最好的聚类数。
2. 没有聚类中心的概念。因此只能在训练集中划分聚类，不能对训练集以外的数据预测其类别归属。
3. 在确定被凝聚的样本时，除了可以以距离作为条件之外，还可以根据连续性来确定被聚集的样本。

相关API：

```python
# 凝聚层次聚类模型
model = sc.AgglomerativeClustering(
    n_clusters=4)
# 凝聚层次不能预测训练集之外的样本，所以
# 直接调用fit_predict
pred_y = model.fit_predict(x)
```

案例：multiple3.txt

```python

```

















