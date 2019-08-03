# 机器学习DAY06

### 总结

回归算法：线性回归、岭回归、多项式回归、决策树回归、正向激励、随机森林、支持向量机回归。

分类算法：逻辑分类、朴素贝叶斯、决策树分类、随机森林、支持向量机分类。

常用的回归算法：岭回归、多项式回归、随机森林、支持向量机

1. 若样本模型比较简单，属于线性模型，则选择岭回归

2. 若岭回归欠拟合，则选择多项式回归

3. 若样本数即特征数量大，则可以选择随机森林
4. SVM支持升维变换，对特征进行扩展，所以样本特征数量不多的情况下也可以使用SVR.

常用的分类算法：逻辑分类、支持向量机分类、朴素贝叶斯、随机森林、

1. 若样本模型比较简单，属于线性模型，逻辑分类
2. SVM支持升维变换，对特征进行扩展，对于分类业务，如果不是特征数特别多，SVM可以适用绝大多数分类场景。
3. 朴素贝叶斯多用于NLP，若已知样本服从某种分布，则可以使用相应的朴素贝叶斯模型训练分类器。
4. 随机森林讲究相似的输入产生相似的输出，适用于数据量大，而且没有特别明显的服从某种概率分布的样本集。

### 聚类

#### 凝聚层次算法

首先假定每个样本都是一个独立的聚类，如果统计出来的聚类数大于期望的聚类数，则从每个样本出发寻找离自己最近的另一个样本，与之聚集，形成更大的聚类，同时令总聚类数减少，不断重复以上过程，直到统计出来的聚类数达到期望值为止。

凝聚层次算法的特点：

1. 聚类数k必须事先已知。借助某些评估指标，优选最好的聚类数。
2. 没有聚类中心的概念，因此只能在训练集中划分聚类，但不能对训练集以外的未知样本确定其聚类归属。不能预测。
3. 在确定被凝聚的样本时，除了以距离作为条件以外，还可以根据连续性来确定被聚集的样本。

凝聚层次算法相关API：

```python
# 凝聚层次聚类器
model = sc.AgglomerativeClustering(n_clusters=4)
pred_y = model.fit_predict(x)
```

案例：重新加载multiple3.txt，使用凝聚层次算法进行聚类划分。 

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_ac.py  凝聚层次
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 基于凝聚层次完成聚类
model = sc.AgglomerativeClustering(
    n_clusters=4)
pred_y = model.fit_predict(x) # 预测点在哪个聚类中
print(pred_y) # 输出每个样本的聚类标签

# 画图显示样本数据
mp.figure('AgglomerativeClustering', facecolor='lightgray')
mp.title('AgglomerativeClustering', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:,0], x[:,1], s=80,
    c=pred_y, cmap='brg', label='Samples')
mp.legend()
mp.show()
```

#### 轮廓系数

好的聚类：内密外疏，同一个聚类内部的样本要足够密集，不同聚类之间样本要足够疏远。

轮廓系数计算规则：针对样本空间中的一个特定样本，计算它与所在聚类其它样本的平均距离a，以及该样本与距离最近的另一个聚类中所有样本的平均距离b，该样本的轮廓系数为(b-a)/max(a, b)，将整个样本空间中所有样本的轮廓系数取算数平均值，作为聚类划分的性能指标s。

轮廓系数的区间为：[-1, 1]。 -1代表分类效果差，1代表分类效果好。0代表聚类重叠，没有很好的划分聚类。

轮廓系数相关API：

```python
import sklearn.metrics as sm
# v：平均轮廓系数
# metric：距离算法：使用欧几里得距离(euclidean)
v = sm.silhouette_score(输入集, 输出集, sample_size=样本数, metric=距离算法)
```

案例：输出KMeans算法聚类划分后的轮廓系数。

```python
# 打印平均轮廓系数
print(sm.silhouette_score( x, pred_y, sample_size=len(x), metric='euclidean'))
```

#### DBSCAN算法

从样本空间中任意选择一个样本，以事先给定的半径做圆，凡被该圆圈中的样本都视为与该样本处于相同的聚类，以这些被圈中的样本为圆心继续做圆，重复以上过程，不断扩大被圈中样本的规模，直到再也没有新的样本加入为止，至此即得到一个聚类。于剩余样本中，重复以上过程，直到耗尽样本空间中的所有样本为止。

DBSCAN算法的特点：

1. 事先给定的半径会影响最后的聚类效果，可以借助轮廓系数选择较优的方案。

2. 根据聚类的形成过程，把样本细分为以下三类：

   外周样本：被其它样本聚集到某个聚类中，但无法再引入新样本的样本。

   孤立样本：聚类中的样本数低于所设定的下限，则不称其为聚类，反之称其为孤立样本。

   核心样本：除了外周样本和孤立样本以外的样本。

DBSCAN聚类算法相关API：

```python
# DBSCAN聚类器
# eps：半径
# min_samples：聚类样本数的下限，若低于该数值，则称为孤立样本
model = sc.DBSCAN(eps=epsilon, min_samples=5)
model.fit(x)
```

案例：修改凝聚层次聚类案例，基于DBSCAN聚类算法进行聚类划分，选择最优半径。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_ literals
"""
demo02_dbscan.py  dbscan聚类
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 读取数据，绘制图像
x = np.loadtxt('../ml_data/multiple3.txt',
    delimiter=',')

# 选择最优半径
epsilons = np.linspace(0.3, 1.0, 8)
models, scores = [], []
for epsilon in epsilons:
    # 针对每个半径构建DBSCAN模型
    model=sc.DBSCAN(eps=epsilon, min_samples=5)
    model.fit(x)
    print(model.labels_)
    score = sm.silhouette_score(x, 
        model.labels_, sample_size=len(x), 
        metric='euclidean')
    models.append(model)
    scores.append(score)
index = np.array(scores).argmax()
best_model = models[index]
best_eps = epsilons[index]
best_score = scores[index]
print(best_eps, best_model, best_score)

# DBSCAN算法的副产品  获取所有核心样本的下标
core_indices=best_model.core_samples_indices
core_mask = np.zeros(len(x), dtype='bool')
core_mask[core_indices] = True
# 孤立样本的掩码
offset_mask = best_model.labels_==-1
# 外周样本的掩码
p_mask = ~(core_mask | offset_mask)

# 画图显示样本数据
mp.figure('DBSCAN', facecolor='lightgray')
mp.title('DBSCAN', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:,0], x[:,1], s=80,
    c=best_model.labels_, cmap='brg', label='Samples')
mp.legend()
mp.show()
```

#### 聚类总结

聚类算法：KMeans，MeanShift，凝聚层次算法，DBSCAN，评估的轮廓系数。

一般聚类都在与“相似度”相关的业务中使用。首先要确定使用什么距离算法实现聚类。聚类算法：欧氏距离，皮尔逊相关距离，曼哈顿距离，余弦距离等。

一般情况下使用KMeans。如果对聚类数量没有要求，需要算法自己判断，则使用均值漂移。如果强调连续性距离的话，则使用凝聚层次。DBSCAN不仅可以聚类，还可以输出孤立样本等副产品。



### 推荐引擎   (用户画像) 

推荐引擎意在把最需要的推荐给用户。

在不同的机器学习场景中通常需要分析相似样本。而统计相似样本的方式可以基于欧氏距离分数，也可基于皮氏距离分数。

**欧氏距离分数**
$$
欧氏距离分数 = \frac{1}{1+欧氏距离}
$$
计算所得欧氏距离分数区间处于：(0, 1]，越趋于0样本间的欧氏距离越远，样本越不相似；越趋于1，样本间的欧氏距离越近，越相似。

构建样本之间的欧氏距离得分矩阵：  
$$
\left[
 \begin{array}{c}
  	  & a & b & c & d & .. \\
  	a & 1 & 0.2 & 0.3 & 0.4 & .. \\
  	b & 0.2 & 1 & x & x & .. \\
  	c & 0.3 & x & 1 & x & .. \\
  	d & 0.4 & x & x & 1 & .. \\
  	.. & .. & .. & .. & .. & .. \\

  \end{array}
  \right]
$$
案例：解析ratings.json，根据每个用户对已观看电影的打分计算样本间的欧氏距离，输出欧氏距离得分矩阵。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_movie.py  电影推荐
"""
import json
import numpy as np

with open('../ml_data/ratings.json', 'r') as f:
    ratings = json.loads(f.read())

# 整理用户之间的相似度得分矩阵
users, scmat = list(ratings.keys()), []
for user1 in users:
    scrow = []
    for user2 in users:
        # 计算user1与user2的相似度 添加到scrow
        movies = set()
        for movie in ratings[user1]:
            if movie in ratings[user2]:
                movies.add(movie)
        if len(movies) == 0:
            score = 0
        else:
            A, B = [], []
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])
            A = np.array(A)
            B = np.array(B)
            # 计算A与B向量的相似度
            score = 1/(1+np.sqrt(((A-B)**2).sum()))

        scrow.append(score)
    scmat.append(scrow)

users = np.array(users)
scmat = np.array(scmat)

for scrow in scmat:
    print(' '.join(['{:.2f}'.format(score)\
             for score in scrow]))
```

**皮尔逊相关系数**

```
A = [1,2,3,1,2] 
B = [3,4,5,3,4] 
m = np.corrcoef(A, B)
```

皮尔逊相关系数 = 协方差 / 标准差之积

相关系数处于[-1, 1]区间。越靠近-1代表两组样本反相关，越靠近1代表两组样本正相关。

案例：使用皮尔逊相关系数计算两用户对一组电影评分的相关性。

```python
score = np.corrcoef(x, y)[0, 1]
```

**按照相似度从高到低排列每个用户的相似用户**

```python
# 按照相似度从高到低排列每个用户的相似用户
for i, user in enumerate(users):
    # 获取所有相似用户得分，去掉自己，排序
    sorted_indices = scmat[i].argsort()[::-1]
    sorted_indices = \
        sorted_indices[sorted_indices != i]
    # user的所有相似用户
    sim_users = users[sorted_indices]
    # user所有相似用户的相似度得分
    sim_scores = scmat[i, sorted_indices]
    print(user, sim_users, sim_scores, sep='\n')
```

**生成推荐清单**

1. 找到所有皮尔逊系数正相关的用户
2. 遍历当前用户的每个相似用户，拿到相似用户看过但是当前用户没有看过的电影作为推荐电影
3. 多个相似用户有可能推荐同一部电影，则取每个相似用户对该电影的评分得均值作为推荐度。

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_movie.py  电影推荐
"""
import json
import numpy as np

with open('../ml_data/ratings.json', 'r') as f:
    ratings = json.loads(f.read())

# 整理用户之间的相似度得分矩阵
users, scmat = list(ratings.keys()), []
for user1 in users:
    scrow = []
    for user2 in users:
        # 计算user1与user2的相似度 添加到scrow
        movies = set()
        for movie in ratings[user1]:
            if movie in ratings[user2]:
                movies.add(movie)
        if len(movies) == 0:
            score = 0
        else:
            A, B = [], []
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])
            A = np.array(A)
            B = np.array(B)
            # 计算A与B向量的相似度
            # score = 1/(1+np.sqrt(((A-B)**2).sum()))
            score = np.corrcoef(A, B)[0,1]

        scrow.append(score)
    scmat.append(scrow)

users = np.array(users)
scmat = np.array(scmat)

for scrow in scmat:
    print(' '.join(['{:.2f}'.format(score)\
             for score in scrow]))

# 按照相似度从高到低排列每个用户的相似用户
for i, user in enumerate(users):
    # 获取所有相似用户得分，去掉自己，排序
    sorted_indices = scmat[i].argsort()[::-1]
    sorted_indices = \
        sorted_indices[sorted_indices != i]
    # user的所有相似用户
    sim_users = users[sorted_indices]
    # user所有相似用户的相似度得分
    sim_scores = scmat[i, sorted_indices]
    # print(user, sim_users, sim_scores, sep='\n')

    # 生成推荐清单
    # 正相关得分的掩码
    positive_mask = sim_scores > 0
    # 获取所有正相关用户的用户名
    sim_users = sim_users[positive_mask]
    # 为user构建推荐清单，找到每个sim_user
    # 看过但当前user没看过的电影，存入字典结构
    # 存储推荐清单：
    # {'电影1':[4.0, 5.0], '电影2':[5.0]}
    reco_movies = {}
    for i, sim_user in enumerate(sim_users):
        for movie, score in \
            ratings[sim_user].items():
            # 相似用户看过，但当前用户没看过
            if movie not in ratings[user].keys():
                if movie not in reco_movies:
                    reco_movies[movie] = [score]
                else:
                    reco_movies[movie].append(score)
    print(user)
    # print(reco_movies)
    # 对推荐清单进行排序
    movie_list = sorted(reco_movies.items(), 
        key=lambda x:np.average(x[1]), 
        reverse=True)
    print(movie_list)
```

### 自然语言处理（NLP）

Siri的工作流程：1. 听  2.懂  3.思考  4.组织语言   5.回答

1. 语音识别
2. 自然语言处理 - 语义分析
3. 逻辑分析 - 结合业务场景与上下文
4. 自然语言处理 - 分析结果生成自然语言文本
5. 语音合成

自然语言处理的常用处理过程：

先针对训练文本进行分词处理（词干提取、原型提取），统计词频，通过词频-逆文档频率算法获得该词对样本语义的贡献，根据每个词的贡献力度，构建有监督分类学习模型。把测试样本交给模型处理，得到测试样本的语义类别。

自然语言工具包 - NLTK

#### 文本分词

分词处理相关API：

```python
import nltk.tokenize as tk
# 把样本按句子进行拆分  sent_list:句子列表
sent_list = tk.sent_tokenize(text)
# 把样本按单词进行拆分  word_list:单词列表
word_list = tk.word_tokenize(text)
#  把样本按单词进行拆分 punctTokenizer：分词器对象
punctTokenizer = tk.WordPunctTokenizer() 
word_list = punctTokenizer.tokenize(text)
```

案例：

```python
import nltk.tokenize as tk
doc = "Are you curious about tokenization? " \
      "Let's see how it works! " \
      "We need to analyze a couple of sentences " \
      "with punctuations to see it in action."
print(doc)	
tokens = tk.sent_tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
print('-' * 15)
tokens = tk.word_tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
print('-' * 15)
tokenizer = tk.WordPunctTokenizer()
tokens = tokenizer.tokenize(doc)
for i, token in enumerate(tokens):
    print("%2d" % (i + 1), token)
```

#### 词干提取

文本样本中的单词的词性与时态对于语义分析并无太大影响，所以需要对单词进行词干提取。

词干提取相关API：

```python
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb

stemmer = pt.PorterStemmer() # 波特词干提取器，偏宽松
stemmer = lc.LancasterStemmer() # 朗卡斯特词干提取器，偏严格
stemmer = sb.SnowballStemmer('english') # 思诺博词干提取器，偏中庸
r = stemmer.stem('playing') # 提取单词playing的词干
```

案例：

```python
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb

words = ['table', 'probably', 'wolves', 'playing',
         'is', 'dog', 'the', 'beaches', 'grounded',
         'dreamt', 'envision']
pt_stemmer = pt.PorterStemmer()
lc_stemmer = lc.LancasterStemmer()
sb_stemmer = sb.SnowballStemmer('english')
for word in words:
    pt_stem = pt_stemmer.stem(word)
    lc_stem = lc_stemmer.stem(word)
    sb_stem = sb_stemmer.stem(word)
    print('%8s %8s %8s %8s' % (
        word, pt_stem, lc_stem, sb_stem))
```

#### 词性还原

与词干提取的作用类似，词性还原更利于人工二次处理。因为有些词干并非正确的单词，人工阅读更麻烦。词性还原可以把名词复数形式恢复为单数形式，动词分词形式恢复为原型形式。

词性还原相关API：

```python
import nltk.stem as ns
# 获取词性还原器对象
lemmatizer = ns.WordNetLemmatizer()
# 把单词word按照名词进行还原
n_lemma = lemmatizer.lemmatize(word, pos='n')
# 把单词word按照动词进行还原
v_lemma = lemmatizer.lemmatize(word, pos='v')
```

案例：

```python
import nltk.stem as ns
words = ['table', 'probably', 'wolves', 'playing',
         'is', 'dog', 'the', 'beaches', 'grounded',
         'dreamt', 'envision']
lemmatizer = ns.WordNetLemmatizer()
for word in words:
    n_lemma = lemmatizer.lemmatize(word, pos='n')
    v_lemma = lemmatizer.lemmatize(word, pos='v')
    print('%8s %8s %8s' % (word, n_lemma, v_lemma))
```

#### 词袋模型

一句话的语义很大程度取决于某个单词出现的次数，所以可以把句子中所有可能出现的单词作为特征名，每一个句子为一个样本，单词在句子中出现的次数为特征值构建数学模型，称为词袋模型。

The brown dog is running. The black dog is in the black room. Running in the room is forbidden.

1 The brown dog is running
2 The black dog is in the black room
3 Running in the room is forbidden

| the  | brown | dog  | is   | running | black | in   | room | forbidden |
| ---- | ----- | ---- | ---- | ------- | ----- | ---- | ---- | --------- |
| 1    | 1     | 1    | 1    | 1       | 0     | 0    | 0    | 0         |
| 2    | 0     | 1    | 1    | 0       | 2     | 1    | 1    | 0         |
| 1    | 0     | 0    | 1    | 1       | 0     | 1    | 1    | 1         |

词袋模型化相关API：

```python
import sklearn.feature_extraction.text as ft

# 构建词袋模型对象
cv = ft.CountVectorizer()
# 训练模型，把句子中所有可能出现的单词作为特征名，每一个句子为一个样本，单词在句子中出现的次数为特征值。
bow = cv.fit_transform(sentences).toarray()
print(bow)
# 获取所有特征名
words = cv.get_feature_names()
```

案例：

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_bow.py  词袋模型
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = "The brown dog is running. " \
	  "The black dog is in the black room. " \
	  "Running in the room is forbidden. "
sents = tk.sent_tokenize(doc)
# 构建词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
print(bow.toarray())
print(cv.get_feature_names())
```

#### 词频（TF）

单词在句子中出现的次数除以句子的总词数称为词频。即一个单词在一个句子中出现的频率。词频相比单词的出现次数可以更加客观的评估单词对一句话的语义的贡献度。词频越高，对语义的贡献度越大。对词袋矩阵归一化即可得到词频。

#### 文档频率（DF）

含有某个单词的文档样本数/总文档样本数。文档频率越高，对语义的贡献度越低。

#### 逆文档频率（IDF）

总样本数/含有某个单词的样本数。逆文档频率越高，对语义的贡献度越高。

#### 词频-逆文档频率(TF-IDF)

词频矩阵中的每一个元素乘以相应单词的逆文档频率，其值越大说明该词对样本语义的贡献越大，根据每个词的贡献力度，构建学习模型。

获取词频逆文档频率（TF-IDF）矩阵相关API：

```python
# 获取词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
# 获取TF-IDF模型训练器
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow).toarray()
```

案例：获取TF_IDF矩阵：

```python
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
print(bow.toarray())
print(cv.get_feature_names())

# 获取TFIDF矩阵
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
print(np.round(tfidf.toarray(), 2))
```

#### 文本分类(主题识别)

使用给定的文本数据集进行主题识别训练，自定义测试集测试模型准确性。

案例：

```python

```

#### 

### 















