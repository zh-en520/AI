# 机器学习DAY04

#### 交叉验证

由于数据集的划分有不确定性，若随机划分的样本正好处于某类特殊样本则得到的训练模型所预测的结果的可信度将受到质疑。

所以需要进行多次交叉验证（Cross Validation），把样本空间中的所有样本均分成n份，使用不同的训练集训练模型，对不同的测试集进行测试，输出最终的指标得分。

```python
import sklearn.model_selection as ms
#获取交叉验证得分
指标值 = ms.cross_val_score(
	model, 输入集， 输出集，
	cv=折叠数,
	scoring='指标名')
# 若指标值达到了要求则：
model.fit(train_x, train_y)
model.predict(...)
```

案例：使用交叉验证输出分类器的精确度

```python
#进行交叉验证，看一下模型精度，若不错再训练
ac = ms.cross_val_score(model, 
	 	train_x, train_y, cv=5, 
	 	scoring='accuracy')
print(ac.mean())
```

**交叉验证的指标**

1. 精确度(accuracy): 分类正确的样本数/总样本数
2. 查准率(precision_weighted): 针对每一个类别，预测正确的样本/预测出来的样本数

3. 召回率(recall_weighted): 针对每一个类别，预测正确的样本数/实际存在的样本数

4. f1得分(f1_weighted): 

   2 x 差准率 x 召回率 / （查准率+召回率）

#### 混淆矩阵

当模型训练完毕后，针对测试集进行测试，可以使用混淆矩阵输出模型的评估结果。混淆矩阵中可以得到每一个类别的查准率、召回率等指标。

```python
import sklearn.metrics as sm
# 使用混淆矩阵评估分类模型的分类效果
m=sm.confusion_matrix(实际输出，预测输出)
```

得到的混淆矩阵的结构：

|       | A类别 | B类别 | C类别 |
| ----- | ----- | ----- | ----- |
| A类别 | 5     | 2     | 1     |
| B类别 | 0     | 6     | 0     |
| C类别 | 1     | 0     | 8     |

查准率 = 主对角线上的值 / 该值所在列之和

召回率 = 主对角线上的值 / 该值所在行之和

每一行和每一列都代表相应类别的样本数量，行代表实际类别，列代表预测类别。如上图：A类别实际有8个样本，预测时5个A类，2个B类，1个C类。B类别预测出来了8个样本，其中实际上2个为A类，6个为B类。

最理想的混淆矩阵结构：

|       | A类别 | B类别 | C类别 |
| ----- | ----- | ----- | ----- |
| A类别 | 5     | 0     | 0     |
| B类别 | 0     | 6     | 0     |
| C类别 | 0     | 0     | 8     |

案例：输出混淆矩阵。

```python
#模型训练
model.fit(train_x, train_y)
#与测试集进行测试，输出混淆矩阵
pred_test_y = model.predict(test_x)
m = sm.confusion_matrix(test_y, pred_test_y)
print(m)
```

#### 分类报告

metrics提供了分类报告相关API，不仅可以得到混淆矩阵，还可以得到交叉验证查准率、召回率、f1得分的结果。

```python
cr = sm.classification_report(
    实际输出，预测输出)
print(cr)
```

#### 决策树分类

决策树分类模型会找到与样本特征匹配的叶子节点，然后以投票的方式进行分类。

案例：基于决策树分类训练模型预测小汽车等级。

```python
"""
demo04_car.py  小汽车评级
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

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
	max_depth=6, n_estimators=200,
	random_state=7)
# 交叉验证
print(ms.cross_val_score(model, 
			train_x, train_y, 
			scoring='f1_weighted').mean())
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
```

#### 验证曲线

验证曲线： 模型性能 = f(超参数)

验证曲线可以选择合适的超参数以使模型性能达到最佳效果。

```python
import sklearn.model_selection as ms

train_scores, test_scores = \
  ms.validation_curve(
	model, 输入集，输出集，
    'n_estimators', # 超参数名称
    [100, 150, 200, ...], # 超参数取值
    cv=5  # 折叠数
)
```

train_scores,test_scores的结构如下：

|      | cv_1  | cv_2  | cv_3  | cv_4  | cv_5  |
| ---- | ----- | ----- | ----- | ----- | ----- |
| 50   | 0.923 | 0.923 | 0.923 | 0.923 | 0.923 |
| 100  | 0.923 | 0.923 | 0.923 | 0.923 | 0.923 |

```python
"""
demo05_vc.py  小汽车评级  验证曲线 max_depth
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
# 使用验证曲线 得到模型最优超参数
train_scores, test_scores = \
	ms.validation_curve(
		model, train_x, train_y,
		'max_depth', 
		np.arange(1, 11), 
		cv=5)
vc_array = np.mean(test_scores, axis=1)
mp.figure('Validation Curve',facecolor='lightgray')
mp.title('Validation Curve', fontsize=16)
mp.xlabel('max_depth')
mp.ylabel('f1_scores')
mp.grid(linestyle=':')
mp.plot(np.arange(1, 11), vc_array)
mp.show()
'''

# 交叉验证
print(ms.cross_val_score(model, 
			train_x, train_y, 
			scoring='f1_weighted').mean())
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
```

#### 学习曲线

学习曲线： 模型性能 = f(训练集大小)

```python
_, train_scores, test_scores = \
  ms.learning_curve(
	model, 输入集，输出集，
    # 训练集大小序列
    train_sizes = [0.9, 0.8, 0.7], 
    cv=5  # 折叠数
)
```

案例：选择最优训练集大小。

```python

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
```

#### 支持向量机 (SVM)

**支持向量机原理**

1. 寻求最优的分类边界

   正确：对大部分样本可以正确的划分类别。

   泛化：最大化支持向量间距。

   公平：类别与支持向量等距。

   简单：线性，直线或平面。

2. 支持基于核函数的升维变换

   通过名为核函数的特征变换，增加新的特征，使得低维度空间中的线性不可分割问题变为高维度空间中的线性可分问题。

**常用核函数**

线性核函数：linear. 该参数不会通过核函数进行维度的提升，仅在原始维度空间中寻求线性分类边界。

```python
import sklearn.svm as svm
model = svm.SVC(kernel='linear')
model.fit(...)
model.predict(...)
```

案例：multiple2.txt 

```python
"""
demo08_svm.py   线性svm
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
model = svm.SVC(kernel='linear')
model.fit(x, y)

# 绘制散点图
# 绘制类别背景  pcolormesh


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
mp.legend()
mp.show()
```

**多项式核函数**

多项式核函数：poly，通过多项式函数增加原始样本特征的高次方程幂，作为新特征。

```python
# 基于svm 实现分类
# degree: 多项式核函数的最高次幂
model = svm.SVC(kernel='poly', 
                degree=2)
model.fit(x, y)
```

**径向基核函数**

径向基核函数：rbf,通过高斯分布函数增加原始样本特征的分布概率作为新的特征。

```python
# 基于svm 实现分类
# gamma: 正态分布曲线的标准差
model = svm.SVC(kernel='rbf', 
                C=600,# 正则强度
                gamma=0.01)
model.fit(x, y)
```

##### 样本类别均衡化

通过类别权重的均衡化，使所占比例较小的样本权重较高，而所占比例较大的样本权重降低，以此平均化不同类别样本对分类模型的贡献，提高模型性能。

```python
svm.SVC(kernel='..',
        class_weighted='balanced')
```

案例：imbalance.txt

```python
# 基于svm 实现分类
model = svm.SVC(kernel='linear',
				class_weight='balanced')
```

##### 置信概率

根据样本与分类边界的距离远近，对其预测类别的可信程度进行量化，离边界越近的样本，置信概率越低，反之，离边界越远的样本，置信概率越高。

获取样本的置信概率：

```python
# 构建模型时，指明需要得到置信概率
model = svm.SVC(... , 
             probability=True)
预测结果 = model.predict(..)
# 对样本矩阵进行预测，
# 并且获取每个样本的置信概率
置信概率 = model.predict_proba(样本矩阵)
```

置信概率矩阵的结构：

|       | 类别1 | 类别2 |
| ----- | ----- | ----- |
| 样本1 | 0.8   | 0.2   |
| 样本2 | 0.9   | 0.1   |
| 样本3 | 1     | 0     |



案例：修改径向基核函数案例，新增样本点，求出每个样本点的置信概率。

```python
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
print(probs)
```















