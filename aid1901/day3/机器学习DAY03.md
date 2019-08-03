# 机器学习DAY03

#### 正向激励

.....

**特征重要性**

作为决策树模型训练过程的副产品，根据每个特征划分子表前后的信息熵减少量就标志了该特征的重要程度，此即为该特征重要性指标。训练得到的模型对象提供了属性：feature_importances_来存储每个特征的重要性。

获取样本矩阵特征重要性属性：

```python
model.fit(train_x, train_y)
fi = model.feature_importances_
```

案例：获取普通决策树与正向激励决策树训练的两个模型的特征重要性值，按照从大到小顺序输出绘图。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_fi.py  特征重要性指标
"""
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pyplot as mp
import numpy as np

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
#获取特征重要性指标
dt_fi = model.feature_importances_
print(dt_fi)

# 基于正向激励模型预测房屋价格
model = se.AdaBoostRegressor(model, 
    n_estimators=400, random_state=7)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
#获取特征重要性指标
ad_fi = model.feature_importances_
print(ad_fi)

# 绘图
mp.figure('Feature Importances', facecolor='lightgray')
mp.subplot(211)
mp.title('Decision Tree FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
names = boston.feature_names
sorted_indices = dt_fi.argsort()[::-1]
x = np.arange(names.size)
mp.bar(x, dt_fi[sorted_indices], 0.8, 
    color='dodgerblue', label='DTFI')
mp.xticks(x, names[sorted_indices])
mp.legend()

mp.subplot(212)
mp.title('AdaBoost FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = ad_fi.argsort()[::-1]
mp.bar(x, ad_fi[sorted_indices], 0.8, 
    color='orangered', label='ADFI')
mp.xticks(x, names[sorted_indices])
mp.legend()

mp.show()
```

##### 自助聚合

每次从总样本矩阵中以有放回抽样的方式随机抽取部分样本构建决策树，这样形成多棵包含不同训练样本的决策树，以削弱某些强势样本对模型预测结果的影响，提高模型的泛化特性。

##### 随机森林

在自助聚合的基础上，每次构建决策树模型时，不仅随机选择部分样本，而且还随机选择部分特征，这样的集合算法，不仅规避了强势样本对预测结果的影响，而且也削弱了强势特征的影响，使模型的预测能力更加泛化。

随机森林相关API：

```python
import sklearn.ensemble as se
# 随机森林回归模型	（属于集合算法的一种）
# max_depth：决策树最大深度10
# n_estimators：构建1000棵决策树，训练模型
# min_samples_split: 子表中最小样本数 若小于这个数字，则不再继续向下拆分
model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
```

案例：分析共享单车的需求，从而判断如何进行共享单车的投放。

```python
'''
1. 读取数据 bike_day.csv
2. 整理输入集 输出集   划分测试集与训练集
3. 选择模型，随机森林，训练模型
4. 使用测试集输出r2得分
5. 输出特征重要性
'''
"""
demo02_bike.py 分析共享单车需求  
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/bike_day.csv',
    unpack=False, dtype='U20', 
    delimiter=',')
print(data.shape)
day_headers = data[0, 2:13] 
x = np.array(data[1:, 2:13], dtype='f8')
y = np.array(data[1:, -1], dtype='f8')

# 划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:], 
# 选择随机森林模型 
model = se.RandomForestRegressor(max_depth=10,
    n_estimators=1000, min_samples_split=3)
# 模型训练
model.fit(train_x, train_y)
# 模型测试
pred_test_y = model.predict(test_x)
# 求r2得分
print(sm.r2_score(test_y, pred_test_y))
# 输出模型的特征重要性
day_fi = model.feature_importances_
mp.figure('Feature Importances', facecolor='lightgray')
mp.subplot(211)
mp.title('Bike_day FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = day_fi.argsort()[::-1]
x = np.arange(day_headers.size)
mp.bar(x, day_fi[sorted_indices], 0.8, 
    color='dodgerblue', label='DTFI')
mp.xticks(x, day_headers[sorted_indices])
mp.legend()

mp.tight_layout()
mp.show()
```

画图显示两组样本数据的特征重要性：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_bike.py 分析共享单车需求  
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = []
with open('../ml_data/bike_day.csv', 'r') as f:
    for line in f.readlines():
        data.append(line.split(','))
f.close()
data = np.array(data)
day_headers = data[0, 2:13] 
x = np.array(data[1:, 2:13], dtype='f8')
y = np.array(data[1:, -1], dtype='f8')

# 划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:], 
# 选择随机森林模型 
model = se.RandomForestRegressor(max_depth=10,
    n_estimators=1000, min_samples_split=3)
# 模型训练
model.fit(train_x, train_y)
# 模型测试
pred_test_y = model.predict(test_x)
# 求r2得分
print(sm.r2_score(test_y, pred_test_y))
# 输出模型的特征重要性
day_fi = model.feature_importances_


data = []
with open('../ml_data/bike_hour.csv', 'r') as f:
    for line in f.readlines():
        data.append(line.split(','))
f.close()
data = np.array(data)
hour_headers = data[0, 2:14] 
x = np.array(data[1:, 2:14], dtype='f8')
y = np.array(data[1:, -1], dtype='f8')

# 划分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:], 
# 选择随机森林模型 
model = se.RandomForestRegressor(max_depth=10,
    n_estimators=1000, min_samples_split=3)
# 模型训练
model.fit(train_x, train_y)
# 模型测试
pred_test_y = model.predict(test_x)
# 求r2得分
print(sm.r2_score(test_y, pred_test_y))
# 输出模型的特征重要性
hour_fi = model.feature_importances_


mp.figure('Feature Importances', facecolor='lightgray')
mp.subplot(211)
mp.title('Bike_day FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = day_fi.argsort()[::-1]
x = np.arange(day_headers.size)
mp.bar(x, day_fi[sorted_indices], 0.8, 
    color='dodgerblue', label='DTFI')
mp.xticks(x, day_headers[sorted_indices])
mp.legend()

mp.subplot(212)
mp.title('Bike_hour FI')
mp.ylabel('Feature Importance')
mp.grid(linestyle=":")
sorted_indices = hour_fi.argsort()[::-1]
x = np.arange(hour_headers.size)
mp.bar(x, hour_fi[sorted_indices], 0.8, 
    color='orangered', label='DTFI')
mp.xticks(x, hour_headers[sorted_indices])
mp.legend()

mp.tight_layout()
mp.show()
```



### 人工分类

| 特征1 | 特征2 | 输出 |
| ----- | ----- | ---- |
| 3     | 1     | 0    |
| 2     | 5     | 1    |
| 1     | 8     | 1    |
| 6     | 4     | 0    |
| 5     | 2     | 0    |
| 3     | 5     | 1    |
| 4     | 7     | 1    |
| 4     | -1    | 0    |
| ...   | ...   | ...  |
| 6     | 8     | 1    |
| 5     | 1     | 0    |

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_sc.py 简单分类
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([[3, 1], 
              [2, 5], 
              [1, 8], 
              [6, 4], 
              [5, 2], 
              [3, 5], 
              [4, 7], 
              [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 根据找到的规律 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
# 当x>y时 样本为0类别  反之则为1类别
grid_z = np.piecewise(grid_x, 
    [grid_x>grid_y, grid_x<grid_y],  [0, 1])
# 绘制样本数据
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification')
mp.xlabel('X')
mp.ylabel('Y')
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:, 0], x[:, 1], s=80,
    c=y, cmap='jet', label='Samples')
mp.legend()
mp.show()
```

### 逻辑分类

通过输入的样本数据，基于多元线型回归模型求出线性预测方程。

y = w<sub>0</sub>+w<sub>1</sub>x<sub>1</sub>+w<sub>2</sub>x<sub>2</sub>

但通过线型回归方程返回的是连续值，不可以直接用于分类业务模型，所以急需一种方式使得把连续的预测值->离散的预测值。   [-oo, +oo]->{0, 1} 
$$
逻辑函数 sigmoid：y = \frac{1}{1+e^{-x}}
$$
该逻辑函数当x>0，y>0.5；当x<0, y<0.5； 可以把样本数据经过线性预测模型求得的值带入逻辑函数的x，即将预测函数的输出看做输入被划分为1类的概率，择概率大的类别作为预测结果，可以根据函数值确定两个分类。这是连续函数离散化的一种方式。

逻辑回归相关API：

```python
import sklearn.linear_model as lm
# 构建逻辑回归器 
# solver：逻辑函数中指数的函数关系（liblinear为线型函数关系）
# C：参数代表正则强度，为了防止过拟合。正则越大拟合效果越小。
model = lm.LogisticRegression(solver='liblinear', C=正则强度)
model.fit(训练输入集，训练输出集)
result = model.predict(带预测输入集)
```

案例：基于逻辑回归器绘制网格化坐标颜色矩阵。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_logisticR.py 逻辑回归
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([[3, 1], 
              [2, 5], 
              [1, 8], 
              [6, 4], 
              [5, 2], 
              [3, 5], 
              [4, 7], 
              [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 根据找到的规律 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
# 构建逻辑回归模型，并训练模型
import sklearn.linear_model as lm
model = lm.LogisticRegression(
    solver='liblinear', C=1)
model.fit(x, y)
# 把网格坐标矩阵中500*500的所有点做类别预测
test_x = np.column_stack(
    (grid_x.ravel(), grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)
# 绘制样本数据
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification')
mp.xlabel('X')
mp.ylabel('Y')
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')

mp.scatter(x[:, 0], x[:, 1], s=80,
    c=y, cmap='jet', label='Samples')

mp.legend()
mp.show()
```

**多元分类**

通过多个二元分类器解决多元分类问题。

| 特征1 | 特征2 | ==>  | 所属类别 |
| ----- | ----- | ---- | -------- |
| 4     | 7     | ==>  | A        |
| 3.5   | 8     | ==>  | A        |
| 1.2   | 1.9   | ==>  | B        |
| 5.4   | 2.2   | ==>  | C        |

若拿到一组新的样本，可以基于二元逻辑分类训练出一个模型判断属于A类别的概率。再使用同样的方法训练出两个模型分别判断属于B、C类型的概率，最终选择概率最高的类别作为新样本的分类结果。

案例：基于逻辑分类模型的多元分类。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_logisticR.py 多元分类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x = np.array([[4, 7], 
              [3.5, 8], 
              [3.1, 6.2], 
              [0.5, 1], 
              [1, 2], 
              [1.2, 1.9], 
              [6, 2], 
              [5.7, 1.5], 
              [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 根据找到的规律 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
# 构建逻辑回归模型，并训练模型
model = lm.LogisticRegression(
    solver='liblinear', C=200)
model.fit(x, y)
# 把网格坐标矩阵中500*500的所有点做类别预测
test_x = np.column_stack(
    (grid_x.ravel(), grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)
# 绘制样本数据
mp.figure('Simple Classification', facecolor='lightgray')
mp.title('Simple Classification')
mp.xlabel('X')
mp.ylabel('Y')
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')

mp.scatter(x[:, 0], x[:, 1], s=80,
    c=y, cmap='jet', label='Samples')

mp.legend()
mp.show()
```

### 朴素贝叶斯分类  

朴素贝叶斯分类是一种依据统计概率理论而实现的一种分类方式。观察这组数据：

| 天气情况  | 穿衣风格  | 约女朋友  | ==>  | 心情      |
| --------- | --------- | --------- | ---- | --------- |
| 0（晴天） | 0（休闲） | 0（约了） | ==>  | 0（高兴） |
| 0         | 1（风骚） | 1（没约） | ==>  | 0         |
| 1（多云） | 1         | 0         | ==>  | 0         |
| 0         | 2（破旧） | 1         | ==>  | 1（郁闷） |
| 2（下雨） | 2         | 0         | ==>  | 0         |
| ...       | ...       | ...       | ==>  | ...       |
| 0         | 1         | 0         | ==>  | ？        |

通过上述训练样本如何预测：晴天、穿着休闲、没有约女朋友时的心情？可以整理相同特征值的样本，计算属于某类别的概率即可。但是如果在样本空间没有完全匹配的数据该如何预测？

**贝叶斯定理：P(A|B)=P(B|A)P(A)/P(B)      <==     *P(A, B) = P(A) P(B|A) = P(B) P(A|B)***       

例如：

假设一个学校里有60%男生和40%女生.女生穿裤子的人数和穿裙子的人数相等,所有男生穿裤子.一个人在远处随机看到了一个穿裤子的学生.那么这个学生是女生的概率是多少?

```
P(女) = 0.4
P(裤子|女) = 0.5
P(裤子) = 0.6 + 0.2 = 0.8
P(女|裤子) = P(裤子|女) * P(女) / P(裤子) = 0.5 * 0.4 / 0.8 = 0.25
```

根据贝叶斯定理，如何预测：晴天、穿着休闲、没有约女朋友时的心情？

```
P(晴天,休闲,没约,高兴) 
= P(晴天|休闲,没约,高兴) P(休闲,没约,高兴) 
= P(晴天|休闲,没约,高兴) P(休闲|没约,高兴) P(没约,高兴)
= P(晴天|休闲,没约,高兴) P(休闲|没约,高兴) P(没约|高兴)P(高兴)
（ 朴素：条件独立，特征值之间没有因果关系）
= P(晴天|高兴) P(休闲|高兴) P(没约|高兴)P(高兴)
```

由此可得，统计总样本空间中晴天、穿着休闲、没有约女朋友时高兴的概率，与晴天、穿着休闲、没有约女朋友时不高兴的概率，择其大者为最终结果。

高斯贝叶斯分类器相关API：

```python
import sklearn.naive_bayes as nb
# 创建高斯分布朴素贝叶斯分类器
model = nb.GaussianNB()
model.fit(x, y)
result = model.predict(samples)
```

案例：multiple1.txt

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_nb.py  朴素贝叶斯分类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb


data = np.loadtxt('../ml_data/multiple1.txt',
    unpack=False, dtype='f8', delimiter=',')
print(data.shape)

x = np.array(data[:, :-1])
y = np.array(data[:, -1])

# 训练NB模型 完成分类业务
model = nb.GaussianNB()
model.fit(x, y)
# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
test_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
test_y = model.predict(test_x)
grid_z = test_y.reshape(grid_x.shape)

# 画图
mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:,0], x[:,1], s=60, c=y, 
	label='Samples', cmap='jet')
mp.legend()
mp.show()
```

#### 



