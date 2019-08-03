# 机器学习DAY15

### 支持向量机(SVM)

#### 支持向量机原理

1. **分类原则：寻求最优分类边界**

   正确：对大部分样本可以正确地划分类别。

   泛化：最大化支持向量间距。

   公平：与支持向量等距。

   简单：线性，直线或平面，分割超平面。

2. **基于核函数的升维变换**

   通过名为核函数的特征变换，增加新的特征，使得低维度空间中的线性不可分问题变为高维度空间中的线性可分问题。 

   **线性核函数**：linear，不通过核函数进行维度提升，仅在原始维度空间中寻求线性分类边界。

   基于线性核函数的SVM分类相关API：

   ```python
   model = svm.SVC(kernel='linear')
   model.fit(train_x, train_y)
   ```

   案例：对multiple2.txt中的数据进行分类。

   ```python
   '''
   1. 读取文件，绘制样本点的分布情况
   2. 拆分测试集训练集
   3. 基于svm训练分类模型
   4. 输出分类效果，绘制分类边界线
   '''
   # -*- coding: utf-8 -*-
   from __future__ import unicode_literals
   """
   demo07_svm.py  svm支持向量机
   """
   import numpy as np
   import sklearn.model_selection as ms
   import sklearn.svm as svm
   import sklearn.metrics as sm
   import matplotlib.pyplot as mp
   
   data = np.loadtxt('../ml_data/multiple2.txt', 
       delimiter=',', dtype='f8')
   x = data[:, :-1]
   y = data[:, -1]
   # 拆分测试集与训练集
   train_x, test_x, train_y, test_y = \
       ms.train_test_split(x, y, test_size=0.25,
       random_state=5)
   # 训练svm模型
   model = svm.SVC(kernel='linear')
   model.fit(train_x, train_y)
   # 预测
   pred_test_y = model.predict(test_x)
   print(sm.classification_report(
           test_y, pred_test_y))
   
   # 绘制分类边界线
   l, r = x[:, 0].min()-1, x[:, 0].max()+1
   b, t = x[:, 1].min()-1, x[:, 1].max()+1
   n = 500
   grid_x, grid_y = np.meshgrid(
       np.linspace(l, r, n), 
       np.linspace(b, t, n))
   bg_x = np.column_stack((grid_x.ravel(), 
       grid_y.ravel()))
   bg_y = model.predict(bg_x)
   grid_z = bg_y.reshape(grid_x.shape)
   
   # 画图显示样本数据
   mp.figure('SVM Classification', facecolor='lightgray')
   mp.title('SVM Classification', fontsize=16)
   mp.xlabel('X', fontsize=14)
   mp.ylabel('Y', fontsize=14)
   mp.tick_params(labelsize=10)
   mp.pcolormesh(grid_x, grid_y, grid_z, 
       cmap='gray')
   mp.scatter(test_x[:,0], test_x[:,1], s=60, 
       c=test_y, label='Samples', cmap='jet')
   mp.legend()
   mp.show()
   ```

   **多项式核函数**：poly，通过多项式函数增加原始样本特征的高次方幂
   $$
   y = x_1+x_2 \\
   y = x_1^2 + 2x_1x_2 + x_2^2 \\
   y = x_1^3 + 3x_1^2x_2 + 3x_1x_2^2 + x_2^3
   $$
   案例，基于多项式核函数训练sample2.txt中的样本数据。

   ```python
   # 基于线性核函数的支持向量机分类器
   model = svm.SVC(kernel='poly', degree=3)
   model.fit(train_x, train_y)
   ```

   **径向基核函数**：rbf，通过高斯分布函数增加原始样本特征的分布概率

   案例，基于径向基核函数训练sample2.txt中的样本数据。

   ```python
   # 基于径向基核函数的支持向量机分类器
   # C：正则强度
   # gamma：正态分布曲线的标准差
   model = svm.SVC(kernel='rbf', C=600, gamma=0.01)
   model.fit(train_x, train_y)
   ```

#### 样本类别均衡化

通过类别权重的均衡化，使所占比例较小的样本权重较高，而所占比例较大的样本权重较低，以此平均化不同类别样本对分类模型的贡献，提高模型性能。

样本类别均衡化相关API：

```python
model = svm.SVC(kernel='linear', class_weight='balanced')
model.fit(train_x, train_y)
```

案例：修改线性核函数的支持向量机案例，基于样本类别均衡化读取imbalance.txt训练模型。

```python
# 训练svm模型
model = svm.SVC(kernel='linear',
    class_weight='balanced')
model.fit(train_x, train_y)
```

#### 置信概率

根据样本与分类边界的距离远近，对其预测类别的可信程度进行量化，离边界越近的样本，置信概率越低，反之，离边界越远的样本，置信概率高。

获取每个样本的置信概率相关API：

```python
# 在获取模型时，给出超参数probability=True
model = svm.SVC(kernel='rbf', C=600, gamma=0.01, probability=True)
预测结果 = model.predict(输入样本矩阵)
# 调用model.predict_proba(样本矩阵)可以获取每个样本的置信概率矩阵
置信概率矩阵 = model.predict_proba(输入样本矩阵)
```

置信概率矩阵格式如下：

|       | 类别1 | 类别2 |
| ----- | ----- | ----- |
| 样本1 | 0.8   | 0.2   |
| 样本2 | 0.9   | 0.1   |
| 样本3 | 0.5   | 0.5   |

案例：修改基于径向基核函数的SVM案例，新增测试样本，输出每个测试样本的置信概率，并给出标注。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_svm_rbf.py  svm支持向量机
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple2.txt', 
    delimiter=',', dtype='f8')
x = data[:, :-1]
y = data[:, -1]
# 拆分测试集与训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=5)
# 训练svm模型
model = svm.SVC(kernel='rbf', C=600, 
    gamma=0.01, probability=True)
model.fit(train_x, train_y)
# 预测
pred_test_y = model.predict(test_x)
print(sm.classification_report(
        test_y, pred_test_y))

# 自定义一组测试样本 输出样本的置信概率
prob_x = np.array([[2, 1.5],
                   [8, 9],
                   [4.8, 5.2],
                   [4, 4],
                   [2.5, 7],
                   [7.6, 2],
                   [5.4, 5.9]])
pred_prob_y = model.predict(prob_x)
probs = model.predict_proba(prob_x)
print(probs)

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

# 画图显示样本数据
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(test_x[:,0], test_x[:,1], s=60, 
    c=test_y, label='Samples', cmap='jet')
mp.scatter(prob_x[:,0], prob_x[:,1], s=80,
    c='orange', label='prob Samples')

# 为每一个点添加备注，标注置信概率
for i in range(len(probs)):
    mp.annotate(
        '[{:.2f}, {:.2f}]'.format(
            probs[i, 0], probs[i, 1]),
        xycoords='data', 
        xy=prob_x[i],
        textcoords='offset points',
        xytext=(-50, 30), fontsize=10,
        color='red',
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='angle3'
        )
    )
mp.legend()
mp.show()
```

#### 网格搜索

获取一个最优超参数的方式可以绘制验证曲线，但是验证曲线只能每次获取一个最优超参数。如果多个超参数有很多排列组合的话，就可以使用网格搜索寻求最优超参数组合。

针对超参数组合列表中的每一个超参数组合，实例化给定的模型，做cv次交叉验证，将其中平均f1得分最高的超参数组合作为最佳选择，实例化模型对象。

网格搜索相关API：

```python
import sklearn.model_selection as ms
model = ms.GridSearchCV(模型, 超参数组合列表, cv=折叠数)
model.fit(输入集，输出集)

# 模型训练的副产品
# 获取网格搜索每个参数组合
model.cv_results_['params']
# 获取网格搜索每个参数组合所对应的平均测试分值
model.cv_results_['mean_test_score']
# 获取最好的参数
model.best_params_
model.best_score_
model.best_estimator_
```

案例：修改置信概率案例，基于网格搜索得到最优超参数。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_gridsearchcv.py  网格搜索
"""
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data = np.loadtxt('../ml_data/multiple2.txt', 
    delimiter=',', dtype='f8')
x = data[:, :-1]
y = data[:, -1]
# 拆分测试集与训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=5)
# 训练svm模型
model = svm.SVC(probability=True)
# 使用网格搜索 寻求最优超参数的组合
params=[{'kernel': ['linear'], 
         'C': [1, 10, 100, 1000]}, 
        {'kernel': ['poly'], 'C':[1], 
         'degree': [2, 3]}, 
        {'kernel': ['rbf'], 
         'C': [1, 10, 100, 1000], 
         'gamma': [1, 0.1, 0.01, 0.001]}]
model = ms.GridSearchCV(model, params, cv=5)
model.fit(train_x, train_y)
# 获取网格搜索的副产品
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_)

# print(model.cv_results_['params'])
# print(model.cv_results_['mean_test_score'])
for p, s in zip(model.cv_results_['params'], 
    model.cv_results_['mean_test_score']):
    print(p, s)


# 预测
pred_test_y = model.predict(test_x)
print(sm.classification_report(
        test_y, pred_test_y))

# 自定义一组测试样本 输出样本的置信概率
prob_x = np.array([[2, 1.5],
                   [8, 9],
                   [4.8, 5.2],
                   [4, 4],
                   [2.5, 7],
                   [7.6, 2],
                   [5.4, 5.9]])
pred_prob_y = model.predict(prob_x)
probs = model.predict_proba(prob_x)
print(probs)

# 绘制分类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

# 画图显示样本数据
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(test_x[:,0], test_x[:,1], s=60, 
    c=test_y, label='Samples', cmap='jet')
mp.scatter(prob_x[:,0], prob_x[:,1], s=80,
    c='orange', label='prob Samples')

# 为每一个点添加备注，标注置信概率
for i in range(len(probs)):
    mp.annotate(
        '[{:.2f}, {:.2f}]'.format(
            probs[i, 0], probs[i, 1]),
        xycoords='data', 
        xy=prob_x[i],
        textcoords='offset points',
        xytext=(-50, 30), fontsize=10,
        color='red',
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='angle3'
        )
    )

mp.legend()
mp.show()
```

### 案例：事件预测

加载event.txt，预测某个时间段是否会出现特殊事件。

案例：

```python
'''
1. 读取文件，加载data数组，删除索引为1的那一列
2. 针对每一列做编码，离散数据使用LabelEncoder，
   连续的数字数据使用DigitEncoder（自定义）
3. 整理数据集，划分测试集训练集
4. 训练SVM分类器，对测试集进行预测
5. 自定义测试数据，实现事件预测
'''
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_event.py  事件预测
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm

class DigitEncoder():
    # 自定义编码器，针对数字字符串做标签编码

    def fit_transform(self, y):
        return y.astype('i4')

    def transform(self, y):
        return y.astype('i4')

    def inverse_transform(self, y):
        return y.astype('str')

data = []
with open('../ml_data/event.txt', 'r')as f:
    for line in f.readlines():
        row = line.split(',')
        row[-1] = row[-1][:-1]
        data.append(row)
f.close()
data = np.array(data)
data = np.delete(data, 1, axis=1)

# 整理输入集与输出集
encoders, x, y = [], [], []
data = data.T
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

# 拆分测试集训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=7)

# 交叉验证
model = svm.SVC(kernel='rbf', 
    class_weight='balanced')
scores = ms.cross_val_score(model, train_x, 
    train_y, cv=5, scoring='f1_weighted')
print(scores.mean())
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print((test_y==pred_test_y).sum()/test_y.size)

# 对测试数据进行测试
data = [['Tuesday', '13:30:00', '21', '23'],
        ['Thursday', '13:30:00', '21', '23']]
data = np.array(data).T
test_x = []
for row in range(len(data)):
    encoder = encoders[row]
    test_x.append(encoder.transform(data[row]))
test_x = np.array(test_x).T

pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(pred_test_y))
```

### 案例：交通流量预测(回归)

加载traffic.txt，预测在某个时间段某个交通路口的车流量。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_traffic.py  车流量预测
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm

class DigitEncoder():
    # 自定义编码器，针对数字字符串做标签编码

    def fit_transform(self, y):
        return y.astype('i4')

    def transform(self, y):
        return y.astype('i4')

    def inverse_transform(self, y):
        return y.astype('str')

data = []
with open('../ml_data/traffic.txt', 'r')as f:
    for line in f.readlines():
        row = line.split(',')
        row[-1] = row[-1][:-1]
        data.append(row)
f.close()
data = np.array(data)

# 整理输入集与输出集
encoders, x, y = [], [], []
data = data.T
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

# 拆分测试集训练集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.25,
    random_state=7)

# 交叉验证
model = svm.SVR(kernel='rbf', 
                C=10, epsilon=0.2)

model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print('r2:', sm.r2_score(test_y, pred_test_y))

# 对测试数据进行测试
data = [['Tuesday', '13:30', 'San Francisco', 'yes'],
        ['Thursday', '13:30', 'San Francisco', 'no']]
data = np.array(data).T
test_x = []
for row in range(len(data)):
    encoder = encoders[row]
    test_x.append(encoder.transform(data[row]))
test_x = np.array(test_x).T

pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(pred_test_y))
```

### 总结

回归算法：线性回归、岭回归、多项式回归、决策树回归、正向激励、随机森林、支持向量机回归。

分类算法：逻辑分类、朴素贝叶斯、决策树分类、随机森林、支持向量机分类。



### 聚类

分类（class）与聚类（cluster）不同，分类是有监督学习模型，聚类属于无监督学习模型。聚类讲究使用一些算法把样本划分为n个群落。一般情况下，这种算法都需要计算欧氏距离。

欧氏距离即欧几里得距离。
$$
P(x_1) - Q(x_2): |x_1-x_2| = \sqrt{(x_1-x_2)^2} \\
P(x_1,y_1) - Q(x_2,y_2): \sqrt{(x_1-x_2)^2+(y_1-y_2)^2} \\
P(x_1,y_1,z_1) - Q(x_2,y_2,z_2): \sqrt{(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2} \\
$$
用两个样本对应特征值之差的平方和之平方根，即欧氏距离，来表示这两个样本的相似性。

#### K均值算法

第一步：随机选择k个样本作为k个聚类的中心，计算每个样本到各个聚类中心的欧氏距离，将该样本分配到与之距离最近的聚类中心所在的类别中。

第二步：根据第一步所得到的聚类划分，分别计算每个聚类的几何中心，将几何中心作为新的聚类中心，重复第一步，直到计算所得几何中心与聚类中心重合或接近重合为止。

**注意：**

1. 聚类数k必须事先已知。借助某些评估指标，优选最好的聚类数。
2. 聚类中心的初始选择会影响到最终聚类划分的结果。初始中心尽量选择距离较远的样本。

K均值算法相关API：

```python
import sklearn.cluster as sc
# n_clusters: 聚类数
model = sc.KMeans(n_clusters=4)
# 不断调整聚类中心，直到最终聚类中心稳定则聚类完成
model.fit(x)
# 获取训练结果的聚类中心
centers = model.cluster_centers_
```

案例：加载multiple3.txt，基于K均值算法完成样本的聚类。

```python
'''
1. 读取文件，加载数据，把样本绘制在窗口中
2. 基于K均值完成聚类业务，为每个样本设置颜色
3. 绘制聚类背景边界线  pcolormesh
'''
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_kmeans.py  k均值
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 基于KMeans完成聚类
model = sc.KMeans(n_clusters=4)
model.fit(x) # 完成聚类
pred_y = model.predict(x) # 预测点在哪个聚类中
print(pred_y) # 输出每个样本的聚类标签
# 获取聚类中心
centers = model.cluster_centers_

# 绘制聚类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

# 画图显示样本数据
mp.figure('KMeans', facecolor='lightgray')
mp.title('KMeans', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:,0], x[:,1], s=80,
    c=pred_y, cmap='brg', label='Samples')
mp.scatter(centers[:,0], centers[:,1],
	color='red',
    s=400, marker='+', label='Cluster Center')
mp.legend()
mp.show()
```

#### 图像量化

KMeans聚类算法可以应用于图像量化领域。通过KMeans算法可以把一张图像所包含的颜色值进行聚类划分，求每一类别的平均值后再重新生成新的图像。可以达到图像降维的目的。这个过程称为图像量化。图像量化可以更好的保留图像的轮廓，降低机器识别图像轮廓的难度。

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_quant.py  图像量化
1. 读取图片的亮度矩阵
2. 基于KMeans算法完成聚类，获取4个聚类中心的值
3. 修改图片，把每个像素亮度值都改为相应类别的均值
4. 绘制
"""
import numpy as np
import scipy.misc as sm
import scipy.ndimage as sn
import sklearn.cluster as sc
import matplotlib.pyplot as mp

img = sm.imread('../ml_data/lily.jpg', True)
# 基于KMeans完成聚类
model = sc.KMeans(n_clusters=4)
x = img.reshape(-1, 1)
model.fit(x)
# 同model.predict(x) 返回每个样本的类别标签
y = model.labels_ 
centers = model.cluster_centers_
img2 = centers[y].reshape(img.shape)

# 绘图
mp.subplot(121)
mp.imshow(img, cmap='gray')
mp.axis('off')  # 关闭坐标轴
mp.subplot(122)
mp.imshow(img2, cmap='gray')
mp.axis('off')  # 关闭坐标轴
mp.tight_layout()
mp.show()
```

#### 均值漂移算法

首先假定样本空间中的每个聚类均服从某种已知的概率分布规则，然后用不同的概率密度函数拟合样本中的统计直方图，不断移动密度函数的中心(均值)的位置，直到获得最佳拟合效果为止。这些概率密度函数的峰值点就是聚类的中心，再根据每个样本距离各个中心的距离，选择最近聚类中心所属的类别作为该样本的类别。

均值漂移算法的特点：

1. 聚类数不必事先已知，算法会自动识别出统计直方图的中心数量。
2. 聚类中心不依据于最初假定，聚类划分的结果相对稳定。
3. 样本空间应该服从某种概率分布规则，否则算法的准确性会大打折扣。

均值漂移算法相关API：

```python
# 量化带宽，决定每次调整概率密度函数的步进量
# n_samples：样本数量
# quantile：量化宽度（直方图一条的宽度）
bw = sc.estimate_bandwidth(x, n_samples=len(x), quantile=0.1)
# 均值漂移聚类器
model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
model.fit(x)
```

案例：加载multiple3.txt，使用均值漂移算法对样本完成聚类划分。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo10_meanshift.py  均值漂移
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 基于MeanShift完成聚类
bw = sc.estimate_bandwidth(x, 
    n_samples=len(x), quantile=0.1)
model = sc.MeanShift(bandwidth=bw, 
    bin_seeding=True)
model.fit(x) # 完成聚类
pred_y = model.predict(x) # 预测点在哪个聚类中
print(pred_y) # 输出每个样本的聚类标签
# 获取聚类中心
centers = model.cluster_centers_

# 绘制聚类边界线
l, r = x[:, 0].min()-1, x[:, 0].max()+1
b, t = x[:, 1].min()-1, x[:, 1].max()+1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), 
    np.linspace(b, t, n))
bg_x = np.column_stack((grid_x.ravel(), 
    grid_y.ravel()))
bg_y = model.predict(bg_x)
grid_z = bg_y.reshape(grid_x.shape)

# 画图显示样本数据
mp.figure('KMeans', facecolor='lightgray')
mp.title('KMeans', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, 
    cmap='gray')
mp.scatter(x[:,0], x[:,1], s=80,
    c=pred_y, cmap='brg', label='Samples')
mp.scatter(centers[:,0], centers[:,1],
	color='red',
    s=400, marker='+', label='Cluster Center')
mp.legend()
mp.show()
```

#### 





