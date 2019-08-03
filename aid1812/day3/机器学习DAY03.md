# 机器学习DAY03

#### 集合算法

#### 正向激励

##### 特征重要性

作为决策树模型训练过程中的副产品，根据每个特征划分子表前后的信息熵减少量来标志该特征的重要程度。此即为该特征重要性指标。

获取样本矩阵特征重要性属性：

```python
model.fit(x, y)
# 训练过后通过feature_importances_
# 获取样本矩阵特征重要性指标
fi = model.feature_importances_
```

案例：获取DT与AdaBoost两种算法不同的特征重要性指标，绘制图像。

```python
"""
demo01_fi.py  特征重要性
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pyplot as mp
import numpy as np

# 加载数据
boston = sd.load_boston()
print(boston.data.shape)  # 样本输入
print(boston.target.shape)# 样本输出
print(boston.feature_names) # 特征名
names = boston.feature_names
'''
 |犯罪率|住宅用地比例|商业用地比例|
 |是否靠河|空气质量|房间数|年限|
 |距市中心距离|路网密度|房产税|师生比|
 |黑人比例|低地位人口比例|
'''
# 打乱原始数据集， 划分训练集与测试集
# 当随机种子相同时得到的随机序列也相同
x, y=su.shuffle(boston.data, boston.target, 
	random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:]

# 构建AdaBoost模型 使用训练集训练，测试集测试
tree=st.DecisionTreeRegressor(max_depth=4)
model = se.AdaBoostRegressor(tree, 
	n_estimators=400, random_state=7)
model.fit(train_x, train_y)
ada_fi = model.feature_importances_
print(ada_fi)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 构建DT模型 使用训练集训练，测试集测试
tree=st.DecisionTreeRegressor(max_depth=4)
tree.fit(train_x, train_y)
tree_fi = tree.feature_importances_
print(tree_fi)
pred_test_y = tree.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 绘制特征重要性的柱状图
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('AdaBoost FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = ada_fi.argsort()[::-1]
pos = np.arange(ada_fi.size)
mp.bar(pos, ada_fi[sorted_inds], 0.8,
	facecolor='dodgerblue', label='AdaBoost')
# 设置刻度文本
mp.xticks(pos, names[sorted_inds]) 
mp.legend()

mp.subplot(212)
mp.title('DT FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = tree_fi.argsort()[::-1]
pos = np.arange(tree_fi.size)
mp.bar(pos, tree_fi[sorted_inds], 0.8,
	facecolor='orangered', label='DT')
# 设置刻度文本
mp.xticks(pos, names[sorted_inds]) 
mp.legend()

mp.tight_layout()
mp.show()
```

#### 自助聚合

每次从总样本矩阵中以有放回的方式随机抽取部分样本构建决策树，这样形成多颗包含不同训练样本的决策树，以削弱某些强势样本对模型预测结果的影响，提高模型的泛化特性。

#### 随机森林

在自助聚合的基础之上，每次构建决策树模型时，不仅随机选择部分样本，而且还随机选择部分特征。这样的集合算法不仅规避了强势样本对预测结果的影响，而且也削弱了强势特征的影响，使模型更加泛化。

```python
import sklearn.ensemble as se
model = se.RandomForestRegressor(
	max_depth=10,# 最大深度
    n_estimators=1000,# 树的数量
    min_samples_split=2#子表中最小样本数
)
```

案例：分析共享单车的需求，判断如何进行共享单车投放。

```python
"""
demo02_rf.py  随机森林实现共享单车需求量预测
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
# 读取数据集

cols = np.arange(2, 16)
data = np.loadtxt('../ml_data/bike_day.csv',
	skiprows=1, unpack=False, 
	delimiter=',', usecols=cols)
day_headers = np.loadtxt('../ml_data/bike_day.csv',
	unpack=False, dtype='U20', delimiter=',')
day_headers = day_headers[0, 2:13]

x = np.array(data[:, 0:11], dtype='f4')
y = np.array(data[:, -1], dtype='f4')
#打乱数据集，划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
print(x.shape, y.shape)
train_size = int(len(x) * 0.9) 
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:], 
# 基于随机森林训练模型
model = se.RandomForestRegressor(
	max_depth=10,
	n_estimators=1000, 
	min_samples_split=2)
model.fit(train_x, train_y)
day_fi = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

data = np.loadtxt('../ml_data/bike_hour.csv',
	unpack=False, dtype='U20',
	delimiter=',')
hour_headers = data[0, 2:14]
x = np.array(data[1:, 2:14], dtype='f4')
y = np.array(data[1:, -1], dtype='f4')
#打乱数据集，划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
print(x.shape, y.shape)
train_size = int(len(x) * 0.9) 
train_x, test_x, train_y, test_y = \
	x[:train_size], x[train_size:], \
	y[:train_size], y[train_size:], 
# 基于随机森林训练模型
model = se.RandomForestRegressor(
	max_depth=10,
	n_estimators=1000, 
	min_samples_split=2)
model.fit(train_x, train_y)
hour_fi = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 绘制两组数据的特征重要性的柱状图 （子图）
# 绘制特征重要性的柱状图
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('RF Day FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = day_fi.argsort()[::-1]
pos = np.arange(day_fi.size)
mp.bar(pos, day_fi[sorted_inds], 0.8,
	facecolor='dodgerblue', label='Day')
# 设置刻度文本
mp.xticks(pos, day_headers[sorted_inds]) 
mp.legend()

mp.subplot(212)
mp.title('RF Hour FI', fontsize=14)
mp.ylabel('importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=":")
sorted_inds = hour_fi.argsort()[::-1]
pos = np.arange(hour_fi.size)
mp.bar(pos, hour_fi[sorted_inds], 0.8,
	facecolor='orangered', label='Hour')
# 设置刻度文本
mp.xticks(pos, hour_headers[sorted_inds]) 
mp.legend()

mp.tight_layout()
mp.show()
```

### 分类问题

#### 人工分类

| 特征 1 | 特征2 | 输出 |
| ------ | ----- | ---- |
| 3      | 1     | 0    |
| 2      | 5     | 1    |
| 1      | 8     | 1    |
| 6      | 4     | 0    |
| 5      | 2     | 0    |
| 3      | 5     | 1    |
| 4      | 7     | 1    |
| 4      | -1    | 0    |
| 6      | 8     | ?    |

案例：在图像中显示样本数据集。

```python
"""
demo03_sc.py  简单人工分类
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([
	[3, 1],
	[2, 5],
	[1, 8],
	[6, 4],
	[5, 2],
	[3, 5],
	[4, 7],
	[4,-1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 把整个空间进行网格化拆分，通过拆分出来的
# 每个点根据分类模型预测每个点类别名，填充
# 相应的颜色值。 pcolormesh
l, r = x[:,0].min()-1, x[:,0].max()+1
b, t = x[:,1].min()-1, x[:,1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
	np.linspace(l, r, n), 
	np.linspace(b, t, n))
grid_z = np.piecewise(grid_x, 
		[grid_x>grid_y, grid_x<=grid_y],
		[0, 1])

mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()
```

#### 逻辑分类

逻辑分类的底层基于线性回归。通过输入的样本数据，基于多元线性回归模型求出线性预测方程：
$$
y = w_0 + w_1x_1 + w_2x_2 +..
$$
通过线性回归方程返回的是连续值，不可以直接用于分类业务模型，所以急需一种方式使得将连续的预测值->离散的预测值. (-∞,+∞) -> {0,1}
$$
逻辑函数 Sigmoid: y = \frac{1}{1+e^{-x}}
$$
该逻辑函数当x>0， y>0.5； 当x<0， y<0.5； 可以把样本数据经过线性预测模型求得的值带入逻辑函数的x，将预测函数的输出看做划分为1类别的概率，由此可以根据函数值确定两个分类。sigmoid函数也是线性函数非线性化的一种方式。

相关API：

```python
import sklearn.linear_model as lm
#liblinear：逻辑函数中的函数关系
model = lm.LogisticRegression(
    solver='liblinear', 
	C=正则强度)
model.fit(...)
model.predict(...)
```

案例：

```python
"""
demo04_lr.py  逻辑分类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x = np.array([
	[3, 1],
	[2, 5],
	[1, 8],
	[6, 4],
	[5, 2],
	[3, 5],
	[4, 7],
	[4,-1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

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
model = lm.LogisticRegression(
		solver='liblinear', C=10)
model.fit(x, y)
# 整理结构变为 25万行2列的二维数组
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
# 整列输出，变维：500*500
grid_z = pred_test_y.reshape(grid_x.shape)

mp.figure('Logistic Classification', facecolor='lightgray')
mp.title('Logistic Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()
```

**多元分类**

通过多个二元分类器解决多元分类问题。

| 特征 1 | 特征2 | 输出 |
| ------ | ----- | ---- |
| 4      | 7     | A    |
| 3.5    | 8     | A    |
| 1.2    | 1.9   | B    |
| 5.4    | 2.2   | C    |

若拿到一组新的样本，可以基于二元逻辑分类训练出一个模型判断属于A类别的概率。再使用相同的算法训练出两个模型分别判断属于B、C类别的概率，最终选择概率最高的类别作为新样本的分类结果。

案例：完成多元分类模型。

```python
"""
demo05_mlr.py  使用逻辑分类解决多元分类问题
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x = np.array([
	[4, 7],
	[3.5, 8],
	[3.1, 6.2],
	[0.5, 1],
	[1, 2],
	[1.2, 1.9],
	[6, 2],
	[5.7, 1.5],
	[5.4, 2.2]])
y = np.array([0,0,0,1,1,1,2,2,2])
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
model = lm.LogisticRegression(
		solver='liblinear', C=1000)
model.fit(x, y)
# 整理结构变为 25万行2列的二维数组
test_x = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
pred_test_y = model.predict(test_x)
# 整列输出，变维：500*500
grid_z = pred_test_y.reshape(grid_x.shape)

mp.figure('Logistic Classification', facecolor='lightgray')
mp.title('Logistic Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()
```

#### 朴素贝叶斯分类

朴素贝叶斯分类是一种依据统计概率理论而实现的一种分类方式。

| 天气情况 | 穿衣风格 | 约女朋友 | ==>  | 心情    |
| -------- | -------- | -------- | ---- | ------- |
| 0(晴天)  | 0(休闲)  | 0(约了)  | ==>  | 0(高兴) |
| 0        | 1(风骚)  | 1(没约)  | ==>  | 0       |
| 1(多云)  | 1        | 0        | ==>  | 0       |
| 0        | 2(破旧)  | 1        | ==>  | 1(郁闷) |
| 2(下雨)  | 2        | 0        | ==>  | 0       |
| ..       | ..       | ..       | ..   | ..      |
| 0        | 1        | 0        |      | ?       |

通过上述训练样本如何预测：晴天、休闲、没约时的心情？根据概率统计学可以求出晴天、休闲、没约时高兴的概率，求出晴天、休闲、没约时郁闷的概率，择其大者为最终的预测结果。

**贝叶斯定理：** 
$$
P(A,B) = P(A)P(B|A) = P(B)P(A|B) \\
\Downarrow \Downarrow \Downarrow\\
P(A|B)=\frac{P(A)P(B|A)}{P(B)}
$$
例如：

假设一个学校里有60%男生和40%女生，女生穿裤子的人数和穿裙子的人数相等。所有男生穿裤子。一个人在远处随机看到了一个穿裤子的学生，问这个学生是女生的概率是多少？

```
P(女) = 0.4
P(裤子|女) = 0.5
P(裤子) = 0.6 + 0.2 = 0.8
P(女|裤子) = 0.4 * 0.5 / 0.8 = 0.25
```

根据贝叶斯定理，求出晴天、休闲、没约、高兴的概率。

```
P(晴天,休闲,没约,高兴)
= P(晴天|休闲,没约,高兴)P(休闲,没约,高兴)
= P(晴天|休闲,没约,高兴)P(休闲|没约,高兴)P(没约,高兴)
= P(晴天|休闲,没约,高兴)P(休闲|没约,高兴)P(没约|高兴)P(高兴)

（朴素：条件独立，特征之间没有因果关系）
= P(晴天|高兴)P(休闲|高兴)P(没约|高兴)P(高兴)
```

由此公式，可以根据统计的方式求出每个概率值，相乘即得到最终的概率结果。

高斯朴素贝叶斯分类器相关API：

```python
import sklearn.naive_bayes as nb
model = nb.GaussianNB()
model.fit(...)
model.predict(...)
```

案例： multiple1.txt

```python
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
# 整列输出，变维：500*500
grid_z = pred_test_y.reshape(grid_x.shape)

mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y,
			cmap='brg', s=80)
mp.show()
```

#### 分类业务数据集的划分

对于分类问题，训练集与测试集的划分不应该用整个样本空间的特定百分比作为训练数据，而应该在其每一个类别的样本中抽取特定百分比划分数据集。sklearn提供了数据集划分相关方法，方便的划分测试与训练数据，使用不同的数据集训练或测试模型，达到提高分类的可信度。

```python
import sklearn.model_selection as ms

训练输入，测试输入，训练输出，测试输出 = 
  ms.train_test_split(
	输入集，输出集，test_size=测试集占比，
	random_state=随机种子)
```

案例：multiple1.txt

```python
"""
demo07_ms.py  数据集的划分
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

# 拆分训练集与测试集
import sklearn.model_selection as ms
train_x, test_x, train_y, test_y = \
	ms.train_test_split(x, y, 
		test_size=0.25, random_state=7)

# 构建NB分类模型
model = nb.GaussianNB()
model.fit(train_x, train_y)
# 整理结构变为 25万行2列的二维数组
grid_xy = np.column_stack(
	(grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(grid_xy)
# 整列输出，变维：500*500
grid_z = grid_z.reshape(grid_x.shape)

mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.pcolormesh(grid_x, grid_y, grid_z,
				cmap='gray')
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y,
			cmap='brg', s=80)
mp.show()
```















