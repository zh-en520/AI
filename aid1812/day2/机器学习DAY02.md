# 机器学习DAY02

#### 线性回归

基于梯度下降理论实现线性回归的步骤：

1. 推导出损失函数
2. 求出w<sub>0</sub> + w<sub>1</sub> 两个方向上的偏导函数，自定义学习率参数使得每次梯度下降迭代时w<sub>0</sub> + w<sub>1</sub> 都在向目标位置进行移动。
3. 不断执行梯度下降，直到找到误差改变量极小（loss函数的极小值）。



绘制随着每次梯度下降，w0，w1，loss的变化曲线。

```python

# 绘制w0，w1，loss的变化曲线
mp.figure('Training Progress', facecolor='lightgray')
mp.subplot(311)
mp.title('Training Progress', fontsize=14)
mp.ylabel('w0', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, w0[:-1], c='dodgerblue',
	label='w0')
mp.legend()

mp.subplot(312)
mp.ylabel('w1', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, w1[:-1], c='dodgerblue',
	label='w1')
mp.legend()

mp.subplot(313)
mp.ylabel('loss', fontsize=12)
mp.tick_params(labelsize=8)
mp.gca().xaxis.set_major_locator(
			mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches, losses, c='dodgerblue',
	label='loss')
mp.legend()
```

基于三维曲面绘制梯度下降过程中的路线。

```python
# 在三维曲面图中绘制梯度下降的路线
import mpl_toolkits.mplot3d as axes3d
grid_w0, grid_w1 = np.meshgrid(
	np.linspace(0, 9, 500),
	np.linspace(0, 3.5, 500))
grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
	grid_loss += \
	 ((grid_w0 + x*grid_w1 - y)**2) / 2

mp.figure('Loss Function')
ax = mp.gca(projection='3d')
ax.set_xlabel('w0', fontsize=12)
ax.set_ylabel('w1', fontsize=12)
ax.set_zlabel('loss', fontsize=12)
ax.plot_surface(grid_w0, grid_w1, 
	grid_loss, rstride=10, cstride=10,
	cmap='jet')
ax.plot(w0[:-1], w1[:-1], losses, 'o-',
	c='red', zorder=3)
mp.tight_layout()
```

以等高线图的方式绘制梯度下降的过程

```python

# 以等高线图的方式绘制梯度下降的过程
mp.figure('Contour', facecolor='lightgray')
mp.title('Contour', fontsize=14)
mp.xlabel('w0', fontsize=12)
mp.ylabel('w1', fontsize=12)
mp.grid(linestyle=':')
mp.contourf(grid_w0, grid_w1, grid_loss,10,
	cmap='jet')
mp.plot(w0, w1, 'o-', c='red', label='BGD')
mp.legend()
```

sklearn提供的线性回归API：

```python
import sklearn.linear_model as lm
# 创建模型对象
model = lm.LinearRegression()
# 模型训练
# 输入：一行一样本，一列一特征的样本矩阵
# 输出：每个样本对应的输出结果
model.fit(输入， 输出)
# 根据输入预测输出
result = model.predict(测试输入)
```

案例：

```python
"""
demo02_lr.py  线性回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/single.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)

# 整理训练集 
x = x.reshape(-1, 1) # x变为n行1列
model = lm.LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)

# 画图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	alpha=0.7, label='Sample Points')
mp.plot(x, pred_y, c='red', linewidth=2,
	label='Regression Line')
mp.legend()
mp.show()
```

##### 评估回归模型训练结果误差（metrics）

回归模型训练完毕后，可以利用测试集评估训练结果误差。sklearn.metrics模块中提供了计算模型误差的常用算法：

```python
import sklearn.metrics as sm
# 平均绝对值误差  np.mean(∑|y-y'|)
sm.mean_absolute_error(y, pred_y)
# 平均平方误差  sqrt(mean(∑(y-y')^2))
sm.mean_squared_error(y, pred_y)
# 中位数绝对值误差  np.median(∑|y-y'|)
sm.median_absolute_error(y, pred_y)
# r2得分(0,1]区间的分值。分值越高，模型越好
sm.r2_score(y, pred_y)
```

#### 模型的保存和加载

模型训练是一个耗时的过程，所以可以在模型训练完毕后把model对象保存到磁盘中。这样的话就可以在需要的时候从磁盘文件中读取出来。直接调用predict使用即可。

```python
import pickle
pickle.dump(model, 磁盘文件f)  #持久化
model = pickle.load(磁盘文件f) #加载对象
```

案例：dump.py   load.py

```python
# 评估训练结果误差
print('r2:', sm.r2_score(y, pred_y))
# 模型存储
with open('../ml_data/linear.pkl', 'wb')as f:
	pickle.dump(model, f)
print('Dump Success!')


# 整理训练集 
with open('../ml_data/linear.pkl', 'rb')as f:
	model = pickle.load(f) 
x = x.reshape(-1, 1)
pred_y = model.predict(x)
```

#### 岭回归

普通线性回归模型使用的是基于最小二乘法实现的回归算法。在最小化损失函数的前提下寻找最优模型参数。 但是在此过程中，若样本空间包含少数异常样本，最终会对模型参数构成不小的影响。使得模型不能更好的对正常样本进行预测。

为此岭回归在模型迭代过程中添加了一个正则项作为超参数，以限制模型参数对异常样本的匹配程度，进而提高模型面对多数正常样本的拟合精度。

```python
import sklearn.linear_model as lm
# 创建岭回归模型对象
model = lm.Ridge(
    正则强度, 
    fit_intercept=True, # 是否训练截距
	max_iter=最大迭代次数 
)
model.fit(...)
mode.predict(...)
```

案例： abnormal.txt

```python
"""
demo05_ridge.py  岭回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)


# 画图
mp.figure('Ridge Regression', facecolor='lightgray')
mp.title('Ridge Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	label='Sample Points')

# 整理训练集 
x = x.reshape(-1, 1) # x变为n行1列
model = lm.Ridge(150, fit_intercept=True, 
	max_iter=1000)
model.fit(x, y)
pred_y = model.predict(x)
mp.plot(x, pred_y, c='orangered', 
	linewidth=2, label='Ridge Regression')

mp.legend()
mp.show()
```

#### 多项式回归

若希望回归模型更好的拟合训练样本数据，可以使用多项式回归模型。

**一元多项式回归**
$$
y=w_0 + w_1x +w_2x^2 + ... + w_dx^d
$$
将高次项看做对一次项特征的扩展，那么上述多项式的一般形式可以理解为：
$$
y=w_0 + w_1x_1 +w_2x_2 + ... + w_dx_d
$$
那么一元多项式回归即可以看做为多元线性回归，可以用LinearRegression模型对样本数据进行模型训练。

所以一元多项式回归的实现需要两个步骤：

1. 将一元多项式回归问题转换为多元线性回归问题。（只需给出最高次的次数即可）
2. 将第一步骤得到的多项式结果中的w1 w2..当做样本特征，交给线性回归器训练多元线性回归模型。

```python
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
# 构建训练一元多项式方程的模型
model = pl.make_pipeline(
    # 多项式特征扩展器 
    # 一元多项式方程 -> 多元线性方程
	sp.PolynomialFeatures(4),
    lm.LinearnRegression()
)
```

案例：single.txt

```python
"""
demo06_poly.py  多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取文件采集数据
x, y = np.loadtxt('../ml_data/single.txt', 
	delimiter=',', usecols=(0,1), 
	unpack=True)

# 画图
mp.figure('Poly Regression', facecolor='lightgray')
mp.title('Poly Regression', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, s=60, marker='o', 
	label='Sample Points')

# 基于管线实现多项式回归
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.metrics as sm
# 构建模型
model = pl.make_pipeline(
	sp.PolynomialFeatures(10), #多项式特征扩展
	lm.LinearRegression())  #线性回归器
x = x.reshape(-1, 1)
model.fit(x, y)
pred_y = model.predict(x)
# r2得分
print('r2:', sm.r2_score(y, pred_y))
# 绘制曲线
x = np.linspace(x.min(), x.max(), 1000)
x = x.reshape(-1, 1)
test_y = model.predict(x)

mp.plot(x, test_y, c='orangered',
	label='Poly Regression')
mp.legend()
mp.show()
```

过于简单的模型，无论对于训练数据还是测试数据都无法给出足够高的预测精度，这种现象称为欠拟合。

过于复杂的模型，对于训练数据可以得到较高的预测精度，但是对于测试数据通常精度较低，这种现象称为过拟合。

一个合适的学习模型应该对训练数据和测试数据都有接近的预测精度，而且精度不能太低。

#### 决策树

**基本算法原理**

核心思想：相似的输入必会产生相似的输出。例如预测薪资.

年龄：1-青年  2-中年  3-老年

学历：1-本科  2-硕士  3-博士

经历：1-出道  2-一般  3-老手  4-骨灰

性别：1-男性  2-女性

| 年龄 | 学历 | 经历 | 性别 | ==>  | 薪资      |
| ---- | ---- | ---- | ---- | ---- | --------- |
| 1    | 1    | 1    | 1    | ==>  | 6000(低)  |
| 2    | 1    | 3    | 1    | ==>  | 10000(中) |
| 3    | 3    | 4    | 1    | ==>  | 50000(高) |
| ...  | ...  | ...  | ...  | ...  | ...       |
| 1    | 3    | 2    | 2    | ==>  | ?         |

为了提高搜索效率，使用树形数据结构处理样本数据：
$$
年龄=1 \left\{
\begin{aligned}
学历1\\
学历2\\
学历3\\
\end{aligned}
\right.
年龄=2 \left\{
\begin{aligned}
学历1\\
学历2\\
学历3\\
\end{aligned}
\right.
年龄=3 \left\{
\begin{aligned}
学历1\\
学历2\\
学历3\\
\end{aligned}
\right.
$$
首先从训练样本矩阵中选择第一个特征进行子表的划分，使每个子表中该特征的值全部相同。然后再在每个子表中选择下一个特征按照同样的规则划分更小的子表，不断重复，直到所有特征全部使用完为止。此时便得到了叶级子表，其中所有的特征值完全相同。

对于待预测样本，根据其每一个特征的值，选择对应的子表，逐一匹配，找到与之完全匹配的叶级子表，用该子表中样本的输出，通过平均(回归问题)或投票(分类)的方式为待预测样本提供输出。

决策树回归器模型相关API：

```python
import sklearn.tree as st
model = st.DecisionTreeRegressor(
    max_depth=4 # 决策树的最大深度
)
model.fit(...)
model.predict(...)
```

案例：预测波士顿地区房屋价格。

```python
"""
demo07_dt.py  决策树 预测波士顿地区房屋价格。
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm

# 加载数据
boston = sd.load_boston()
print(boston.data.shape)  # 样本输入
print(boston.target.shape)# 样本输出
print(boston.feature_names) # 特征名
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

# 构建模型 使用训练集训练，测试集测试
model=st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
```

**工程优化**

某些情况不必用尽所有的特征，叶级子表中允许混杂不同的特征值，以此可以降低决策树的层数。在精度牺牲可以接收的前提下，提高模型的性能。

通常情况下，可以优先选择使**信息熵(数据混杂程度。信息熵也大，数据越混杂，信息熵越小，数据越纯净)**减少量最大的特征作为划分子表的依据。

#### 集合算法

根据多个不同的模型给出的预测结果，利用平均（回归）或投票（分类）的方法，得出最终预测的结果。

基于决策树的集合算法，就是按照某种规则，构建多棵彼此不同的决策树模型，分别给出针对未知样本的预测结果，最后通过平均或投票的方式得到相对综合的结论。

##### 正向激励

首先为样本矩阵中的样本随机分配初始权重，由此构成一个带有权重的决策树。再由该决策数提供预测输出时，通过加权平均或者加权投票的方式产生预测值。

将训练样本带入模型，预测其输出，对于那些预测值与实际值不同的样本，提高其权重，由此形成第二棵决策树，重复以上过程，构建不同权重的若干颗决策树。

相关API：

```python
import sklearn.tree as st
import sklearn.ensemble as se
# 构建决策树
tree = st.DecisionTreeRegressor(...)
# 构建正向激励模型
model = se.AdaBoostRegressor(
    tree, # 基础模型
	n_estimators=400, # 构建400棵决策树
	random_state=7 # 随机种子
)
model.fit(...)
model.predict(...)
```

案例：基于正向激励训练boston房价。

```python
"""
demo07_adaboost.py  正向激励
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se

# 加载数据
boston = sd.load_boston()
print(boston.data.shape)  # 样本输入
print(boston.target.shape)# 样本输出
print(boston.feature_names) # 特征名
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

# 构建模型 使用训练集训练，测试集测试
tree=st.DecisionTreeRegressor(max_depth=4)
model = se.AdaBoostRegressor(tree, 
	n_estimators=400, random_state=7)
model.fit(train_x, train_y)

pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

pred_train_y = model.predict(train_x)
print(sm.r2_score(train_y, pred_train_y))
```









