# 机器学习DAY02

### 线性回归

案例：画图模拟梯度下降的过程

1. 整理训练集数据，自定义梯度下降算法规则，求出w<sub>0</sub> ， w<sub>1</sub> ，绘制回归线。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_lr.py  梯度下降实现线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
# 实现梯度下降过程
times = 1000
lrate = 0.01
w0, w1 = [1], [1] # 记录每次梯度下降的参数
losses = [] # 保存每次迭代损失函数值
epoches = []  # 保存每次迭代过程的索引
for i in range(1, times+1):
    # 输出每次下降时：w0 w1 loss值的变化
    epoches.append(i)
    loss = ((w0[-1] + w1[-1]*train_x \
             - train_y)**2).sum() / 2
    losses.append(loss)
    print('{:4}> w0={:.6f}, w1={:.6f}, ' \
          'loss={:.6f}'.format(
          epoches[-1], w0[-1], w1[-1],
          losses[-1]))

    # 每次梯度下降过程需要求出w0与w1的修正值
    # 求修正值需要推导loss函数在w0及w1方向的偏导数
    d0=(w0[-1] + w1[-1]*train_x - train_y).sum()
    d1=((w0[-1] + w1[-1]*train_x - train_y)\
        * train_x).sum()
    w0.append(w0[-1] - lrate*d0)
    w1.append(w1[-1] - lrate*d1)

print(w0[-1], w1[-1])
pred_y = w0[-1] + w1[-1]*train_x

# 绘制样本点
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression')
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, s=60, 
    marker='o', c='orangered', label='Samples')
# 绘制回归线
mp.plot(train_x, pred_y, color='dodgerblue',
    label='Regression Line')

mp.legend()
mp.show()
```

1. 绘制随着每次梯度下降，w<sub>0</sub>，w<sub>1</sub>，loss的变化曲线。

```python
#绘制随着每次梯度下降，w0，w1，loss的变化曲线。
mp.figure('BGD Params', facecolor='lightgray')
mp.title('BGD Params')
mp.tick_params(labelsize=8)
mp.subplot(311)
mp.grid(linestyle=':')
mp.plot(epoches, w0[:-1], color='dodgerblue',
    label='w0')
mp.legend()
mp.subplot(312)
mp.grid(linestyle=':')
mp.plot(epoches, w1[:-1], color='dodgerblue',
    label='w1')
mp.legend()
mp.subplot(313)
mp.grid(linestyle=':')
mp.plot(epoches, losses, color='dodgerblue',
    label='loss')
mp.legend()
mp.tight_layout()
```

1. 基于三维曲面绘制梯度下降过程中的每一个点。

```python
# 基于三维曲面绘制梯度下降过程中的每一个点。
import mpl_toolkits.mplot3d as axes3d
# 整理网格点坐标矩阵，计算每个点的loss绘制曲面
grid_w0, grid_w1 = np.meshgrid(
    np.linspace(0, 9, 500),
    np.linspace(0, 3.5, 500))
grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
    grid_loss += \
        ((grid_w0 + grid_w1*x - y)**2)/2
mp.figure('Loss Function',facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(grid_w0, grid_w1, 
    grid_loss, cmap='jet')
ax3d.plot(w0[:-1], w1[:-1], losses, 
    'o-', c='orangered', label='BGD')
mp.tight_layout()
```

1. 以等高线的方式绘制梯度下降的过程。

```python
# 以等高线的方式绘制梯度下降的过程。
mp.figure('BGD Contour', facecolor='lightgray')
mp.title('BGD Contour')
mp.xlabel('w0')
mp.ylabel('w1')
mp.grid(linestyle=':')
cntr=mp.contour(grid_w0, grid_w1, grid_loss,
    c='black', linewidths=0.5)
mp.clabel(cntr, fmt='%.2f', 
    inline_spacing=0.2, fontsize=8)
mp.contourf(grid_w0, grid_w1, grid_loss,
    cmap='jet')
mp.plot(w0[:-1], w1[:-1], 'o-',
    c='orangered', label='BGD')
```



**线性回归相关API：**

```python
import sklearn.linear_model as lm
# 创建模型
model = lm.LinearRegression()
# 训练模型
# 输入为一个二维数组表示的样本矩阵
# 输出为每个样本最终的结果
model.fit(输入, 输出) # 训练模型
# 预测输出 
# 输入array是一个二维数组，每一行是一个样本，每一列是一个特征。
result = model.predict(array)
```

案例：基于线性回归训练single.txt中的训练样本，使用模型预测测试样本。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_lr.py  线性回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 把输入变为二维数组，一行一样本，一列一特征
x = x.reshape(-1, 1) # 变为n行1列
print(x.shape)
model = lm.LinearRegression()
model.fit(x, y)
# 把样本x带入模型求出预测y
pred_y = model.predict(x)

# 绘制图像
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')
mp.scatter(x, y, c='dodgerblue', s=80,
    marker='o', label='Points')
mp.plot(x, pred_y, c='orangered',
    label='Regression Line')
mp.legend()
mp.show()
```

#### 评估训练结果误差（metrics）

线性回归模型训练完毕后，可以利用测试集评估训练结果误差。sklearn.metrics提供了计算模型误差的几个常用算法：

```python
import sklearn.metrics as sm

# 平均绝对值误差：1/m∑|实际输出-预测输出|
sm.mean_absolute_error(y, pred_y)
# 平均平方误差：SQRT(1/mΣ(实际输出-预测输出)^2)
sm.mean_squared_error(y, pred_y)
# 中位绝对值误差：MEDIAN(|实际输出-预测输出|)
sm.median_absolute_error(y, pred_y)
# R2得分，(0,1]区间的分值。分数越高，误差越小。
sm.r2_score(y, pred_y)
```

案例：在上一个案例中使用sm评估模型误差。

```python
# 平均绝对值误差：1/m∑|实际输出-预测输出|
print(sm.mean_absolute_error(y, pred_y))
# 平均平方误差：SQRT(1/mΣ(实际输出-预测输 出)^2)
print(sm.mean_squared_error(y, pred_y))
# 中位绝对值误差：MEDIAN(|实际输出-预测输出|)
print(sm.median_absolute_error(y, pred_y))
# R2得分，(0,1]区间的分值。分数越高，误差越小。
print(sm.r2_score(y, pred_y))
```

#### 模型的保存和加载

模型训练是一个耗时的过程，一个优秀的机器学习模型是非常宝贵的。可以模型保存到磁盘中，也可以在需要使用的时候从磁盘中重新加载模型即可。不需要重新训练。

模型保存和加载相关API：

```python
import pickle
pickle.dump(model, 磁盘文件) # 保存模型
model = pickle.load(磁盘文件)  # 加载模型
```

案例：把训练好的模型保存到磁盘中。

```python
# 将训练好的模型对象保存到磁盘文件中
with open('../../data/linear.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# 从磁盘文件中加载模型对象
with open('../../data/linear.pkl', 'rb') as f:
    model = pickle.load(f)
# 根据输入预测输出
pred_y = model.predict(x)
```

#### 岭回归

普通线性回归模型使用基于梯度下降的最小二乘法，在最小化损失函数的前提下，寻找最优模型参数，于此过程中，包括少数异常样本在内的全部训练数据都会对最终模型参数造成程度相等的影响，异常值对模型所带来影响无法在训练过程中被识别出来。为此，岭回归在模型迭代过程所依据的损失函数中增加了正则项，以限制模型参数对异常样本的匹配程度，进而提高模型面对多数正常样本的拟合精度。

```python
import sklearn.linear_model as lm
# 创建模型
model = lm.Ridge(正则强度，fit_intercept=是否训练截距, max_iter=最大迭代次数)
# 训练模型
# 输入为一个二维数组表示的样本矩阵
# 输出为每个样本最终的结果
model.fit(输入, 输出)
# 预测输出  
# 输入array是一个二维数组，每一行是一个样本，每一列是一个特征。
result = model.predict(array)
```

案例：加载abnormal.txt文件中的数据，基于岭回归算法训练回归模型。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_ridge.py  岭回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 采集数据
x, y = np.loadtxt('../ml_data/abnormal.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 把输入变为二维数组，一行一样本，一列一特征
x = x.reshape(-1, 1) # 变为n行1列
model = lm.Ridge(200, fit_intercept=True,
    max_iter=1000)
model.fit(x, y)
# 把样本x带入模型求出预测y
pred_y = model.predict(x)

# 输出模型的评估指标  sm
import sklearn.metrics as sm
print(sm.mean_absolute_error(y, pred_y))
print(sm.mean_squared_error(y, pred_y))
print(sm.median_absolute_error(y, pred_y))
print(sm.r2_score(y, pred_y))

# 绘制图像
mp.figure('Ridge Regression', facecolor='lightgray')
mp.title('Ridge Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')
mp.scatter(x, y, c='dodgerblue', s=80,
    marker='o', label='Points')
mp.plot(x, pred_y, c='orangered',
    label='Regression Line')
mp.legend()
mp.show()
```

### 多项式回归

若希望回归模型更好的拟合训练样本数据，可以使用多项式回归器。

**一元多项式回归**

y=w<sub>0</sub> + w<sub>1</sub> x + w<sub>2</sub> x<sup>2</sup> + w<sub>3</sub> x<sup>3</sup> + ... + w<sub>d</sub> x<sup>d</sup>

将高次项看做对一次项特征的扩展得到：

y=w<sub>0</sub> + w<sub>1</sub> x<sub>1</sub>  + w<sub>2</sub> x<sub>2</sub>  + w<sub>3</sub> x<sub>3</sub>  + ... + w<sub>d</sub> x<sub>d</sub> 

那么一元多项式回归即可以看做为多元线性回归，可以使用LinearRegression模型对样本数据进行模型训练。

所以一元多项式回归的实现需要两个步骤：

1. 将一元多项式回归问题转换为多元线性回归问题（只需给出多项式最高次数即可）。
2. 将1步骤得到多项式的结果中 w<sub>1</sub>  w<sub>2</sub>  .. 当做样本特征，交给线性回归器训练多元线性模型。

使用sklearn提供的**数据管线**实现两个步骤的顺序执行：

```python
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm

model = pl.make_pipeline(
    # 10: 多项式的最高次数
    sp.PolynomialFeatures(10),  # 多项式特征扩展器
    lm.LinearRegression())      # 线性回归器
```

案例：single.txt

```python
'''
1. 导包
2. 读取数据
3. 创建多项式回归模型   （同API）
4. 模型训练及预测
5. 通过模型预测得到的pred_y，绘制多项式函数图像
'''
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_poly.py  多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import sklearn.metrics as sm

# 采集数据
x, y = np.loadtxt('../ml_data/single.txt',
    delimiter=',', usecols=(0, 1),
    unpack=True)

# 创建模型
x = x.reshape(-1, 1)
model = pl.make_pipeline(
    sp.PolynomialFeatures(10),
    lm.LinearRegression())
model.fit(x, y)
# 针对所有的x  求预测值pred_y
pred_y = model.predict(x)
print(sm.r2_score(y, pred_y))

# 绘制多项式回归线
px = np.linspace(x.min(), x.max(), 1000)
px = px.reshape(-1, 1)
pred_py = model.predict(px)

# 绘制图像
mp.figure('Poly Regression', facecolor='lightgray')
mp.title('Poly Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')
mp.scatter(x, y, c='dodgerblue', s=80,
    marker='o', label='Points')
mp.plot(px, pred_py, c='orangered', 
    label='PolyFit Line')
mp.legend()
mp.show()
```

过于简单的模型，无论对于训练数据还是测试数据都无法给出足够高的预测精度，这种现象叫做欠拟合。

过于复杂的模型，对于训练数据可以得到较高的预测精度，但对于测试数据通常精度较低，这种现象叫做过拟合。

一个性能可以接受的学习模型应该对训练数据和测试数据都有接近的预测精度，而且精度不能太低。

```
训练集R2   测试集R2
0.3        0.4    欠拟合：过于简单，无法反映数据的规则
0.9        0.2    过拟合：过于复杂，太特殊，缺乏一般性
0.7        0.6    可接受：复杂度适中，既反映数据的规则，同时又不失一般性
```

### 决策树

#### 基本算法原理

核心思想：相似的输入必会产生相似的输出。例如预测某人薪资：

年龄：1-青年，2-中年，3-老年
学历：1-本科，2-硕士，3-博士
经历：1-出道，2-一般，3-老手，4-骨灰
性别：1-男性，2-女性

| 年龄 | 学历 | 经历 | 性别 | ==>  | 薪资        |
| ---- | ---- | ---- | ---- | ---- | ----------- |
| 1    | 1    | 1    | 1    | ==>  | 6000（低）  |
| 2    | 1    | 3    | 1    | ==>  | 10000（中） |
| 3    | 3    | 4    | 1    | ==>  | 50000（高） |
| ...  | ...  | ...  | ...  | ==>  | ...         |
| 1    | 3    | 2    | 2    | ==>  | ?           |

为了提高搜索效率，使用树形数据结构处理样本数据：
$$
年龄=1\left\{
\begin{aligned}
学历1 \\
学历2 \\
学历3 \\
\end{aligned}
\right.
\quad\quad
年龄=2\left\{
\begin{aligned}
学历1 \\
学历2 \\
学历3 \\
\end{aligned}
\right.
\quad\quad
年龄=3\left\{
\begin{aligned}
学历1 \\
学历2 \\
学历3 \\
\end{aligned}
\right.
$$
首先从训练样本矩阵中选择第一个特征进行子表划分，使每个子表中该特征的值全部相同，然后再在每个子表中选择下一个特征按照同样的规则继续划分更小的子表，不断重复直到所有的特征全部使用完为止，此时便得到叶级子表，其中所有样本的特征值全部相同。对于待预测样本，根据其每一个特征的值，选择对应的子表，逐一匹配，直到找到与之完全匹配的叶级子表，用该子表中样本的输出，通过平均(回归)或者投票(分类)为待预测样本提供输出。

随着子表的划分，信息熵（信息的混乱程度）越来越小，信息越来越纯，数据越来越有序。

决策树回归器模型相关API：

```python
import sklearn.tree as st

# 创建决策树回归器模型  决策树的最大深度为4
model = st.DecisionTreeRegressor(max_depth=4)
# 训练模型  
# train_x： 二维数组样本数据
# train_y： 训练集中对应每行样本的结果
model.fit(train_x, train_y)
# 测试模型
pred_test_y = model.predict(test_x)
```

案例：预测波士顿地区房屋价格。

读取数据，打乱原始数据集，划分测试与训练集。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_boston.py   波士顿房屋价格预测
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm

# 加载波士顿地区房屋价格
boston = sd.load_boston()
# ['犯罪率' '住宅地比例' '商业用地比例' 
#  '是否靠河' '空气质量' '房间数' '年限' 
#  '距市中心的距离' '路网密度' '房产税' 
#  '师生比' '黑人比例' '低地位人口比例']

print(boston.feature_names) # 特征名
print(boston.data.shape) # 数据的输入
print(boston.target.shape)  # 数据的输出

# 划分测试集与训练集    80%做训练
# random_state 若打乱时使用的随机种子相同，
# 则得到的结果相同。
x, y = su.shuffle(boston.data, 
    boston.target, random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]
print(train_x.shape)
print(test_x.shape)

# 基于决策树建模->训练模型->测试模型
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
```

#### 工程优化

不必用尽所有的特征，叶级子表中允许混杂不同的特征值，以此降低决策树的层数，在精度牺牲可接受的前提下，提高模型的性能。通常情况下，可以优先选择使<b style='color:red;'>信息熵</b>减少量最大的特征作为划分子表的依据。

#### 集合算法

根据多个不同模型给出的预测结果，利用平均(回归)或者投票(分类)的方法，得出最终预测结果。

基于决策树的集合算法，就是按照某种规则，构建多棵彼此不同的决策树模型，分别给出针对未知样本的预测结果，最后通过平均或投票得到相对综合的结论。

##### 正向激励

首先为样本矩阵中的样本随机分配初始权重，由此构建一棵带有权重的决策树，在由该决策树提供预测输出时，通过加权平均或者加权投票的方式产生预测值。将训练样本代入模型，预测其输出，对那些预测值与实际值不同的样本，提高其权重，由此形成第二棵决策树。重复以上过程，构建出不同权重的若干棵决策树。

正向激励相关API：

```python
import sklearn.tree as st
import sklearn.ensemble as se
# model: 决策树模型（一颗）
model = st.DecisionTreeRegressor(max_depth=4)
# 自适应增强决策树回归模型	
# n_estimators：构建400棵不同权重的决策树，训练模型
model = se.AdaBoostRegressor(model, n_estimators=400, random_state=7)
# 训练模型
model.fit(train_x, train_y)
# 测试模型
pred_test_y = model.predict(test_x)
```

案例：基于正向激励训练预测波士顿地区房屋价格的模型。

```python
# 基于正向激励模型预测房屋价格
import sklearn.ensemble as se

model = se.AdaBoostRegressor(model, 
    n_estimators=400, random_state=7)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
```













