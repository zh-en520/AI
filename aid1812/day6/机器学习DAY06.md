# 机器学习DAY06

#### 凝聚层次算法

首先假定每个样本都是一个独立的聚类，如果统计出来的聚类数大于期望的聚类数，则从每个样本出
发，寻找离自己最近的另一个样本，与之聚集，形成更大的聚类，同时令总聚类数减少，不断重复以上过程，直到统计出来的聚类数达到期望为止。

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

有连续性的凝聚层次聚类器：

```python
import sklearn.neighbors as nb

conn = nb.kneighbors_graph(
    x, 10, include_self=False)
sc.AgglomerativeClustering(
    n_clusters=4,
	linkage='average',
	connectivity=conn)
```

#### 轮廓系数

好的聚类：内密外疏，同一个聚类内部的样本要足够密集，不同聚类之间的样本要足够疏远。

轮廓系数的计算规则：针对样本空间中的一个特定样本，计算它所在聚类其他样本的平均距离a，以及该样本与距离最近的另一个聚类中所有样本的平均距离b，该样本的轮廓系数：(b-a)/max(a,b)。将整个样本空间中所有样本的轮廓系数取平均值作为聚类划分的性能评估指标。

轮廓系数：[-1,1]。 -1代表聚类效果差，1代表聚类效果好。0代表聚类重叠。

```python
import sklearn.metrics as sm
v = sm.silhouette_score(
	输入集，输出集，
    sample_size=样本数量,
	metric=距离算法)
print(v)
```

案例，测试KMean聚类的轮廓系数。

```python
# euclidean：欧氏距离
v = sm.silhouette_score(
	x, pred_y, sample_size=len(x),
	metric='euclidean')
print(v)
```

#### DBSCAN算法

从样本空间中任意选择一个样本，以事先给定的半径做圆，凡是被圆圈中的样本都视为与当前样本处于同一个聚类。以这些被圈中的样本为圆心继续做圆，重复以上过程，不断扩大被圈中的样本规模。直到再也没有新的样本加入为止，至此得到一个聚类。在剩余样本中，重复以上过程，直到耗尽所有样本为止。

DBSCAN算法的特点：

1. 事先给定的半径会影响最后的聚类效果，可以借助轮廓系数优选参数。

2. 根据聚类的形成过程，该算法支持把样本分为以下三类：

   外周样本：被其他样本聚集到某个聚类中，但是无法再引入新样本的样本。

   孤立样本：聚类中的样本数低于所设定的下限，则不称其为聚类，称为孤立样本。

   核心样本：除了外周样本和孤立样本之外的样本。

```python
# eps：epsilon 圆的半径
# min_samples: 聚类数量的下限，低于该数量
# 将成为孤立样本。
model = sc.DBSCAN(
    eps=epsilon, min_samples=3)
model.fit(x)
```

案例：基于DBSAN算法进行聚类划分，选择最优半径。

```python
"""
demo03_dbscan.py  dbscan
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

x = np.loadtxt('../ml_data/perf.txt',
	delimiter=',')
# 优选参数
epsilons, scores, models = \
	np.linspace(0.3, 1.2, 10), [], []
for epsilon in epsilons:
	model=sc.DBSCAN(eps=epsilon, min_samples=5)
	model.fit(x)
	score = sm.silhouette_score(
		x, model.labels_, sample_size=len(x), 
		metric='euclidean')
	scores.append(score)
	models.append(model)

scores = np.array(scores)
best_index = scores.argmax()
print('best eps:', epsilons[best_index])
print('best model:', models[best_index])
print('best scores:', scores[best_index])

# 获取核心样本、外周样本、孤立样本，并且绘图
besk_model = models[best_index]
pred_y = besk_model.fit_predict(x)
# 获取核心样本的掩码
core_mask = np.zeros(len(x), dtype=bool)
core_mask[besk_model.core_sample_indices_] = True
# 孤立样本的掩码
offset_mask = pred_y == -1
# 外周样本的掩码
periphery_mask = ~(core_mask|offset_mask)

# 画图
mp.figure('DBSCAN Clustering', facecolor='lightgray')
mp.title('DBSCAN Clustering', fontsize=14)
mp.xlabel('x', fontsize=12)
mp.ylabel('y', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x[core_mask][:,0], 
	x[core_mask][:,1], c=pred_y[core_mask],
	s=80, label='Core', cmap='brg')
mp.scatter(x[periphery_mask][:,0], 
	x[periphery_mask][:,1], alpha=0.5,
	c=pred_y[periphery_mask],
	s=70, label='periphery', cmap='brg')
mp.scatter(x[offset_mask][:,0], 
	x[offset_mask][:,1], 
	c='gray',label='offset')

mp.legend()
mp.show()
```

#### 推荐引擎

推荐引擎的作用在于最需要的内容推荐给用户。

在不同的推荐场景中通常需要分析相似样本。而统计相似样本的方式可以基于欧氏距离分数，也可以基于皮氏距离分数。

**欧氏距离分数**
$$
欧氏距离分数 = \frac{1}{1+欧氏距离}
$$
欧氏距离分数区间处于：(0, 1]， 越趋近于0两样本的距离越远，越趋近于1，两样本的距离越近。

构建样本之间的欧氏距离得分矩阵：

|      | a    | b    | c    | d    | ...  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| a    | 1    | 0.2  | 0.3  | 0.4  | ...  |
| b    | 0.2  | 0.3  | 0.6  | 0.9  | ...  |
| c    | 0.3  | 0.6  | 0.4  | 0.4  | ...  |
| d    | 0.4  | 0.9  | 0.4  | 0.4  | ...  |
| ...  | ...  | ...  | ...  | ...  | ...  |

案例：解析ratings.json, 根据每个用户对已观看电影的打分，计算样本间的欧氏距离，输出欧氏距离得分矩阵。

```python
"""
demo04_reco.py 推荐引擎
"""
import json
import numpy as np

with open(
	'../ml_data/ratings.json', 'r') as f:
	ratings = json.loads(f.read())

# scmat存储相似度得分矩阵
users, scmat = list(ratings.keys()), []

for user1 in users:
	scrow = []
	for user2 in users:
		movies = set()
		# 把两个人都看过的电影存入movies中
		for movie in ratings[user1]:
			if movie in ratings[user2]:
				movies.add(movie)
		# 二人没有共同语言
		if len(movies) == 0:
			score = 0
		else: 
			# 分别存储两人对同一部电影的打分
			x1, x2 = [], []
			for movie in movies:
				x1.append(ratings[user1][movie])
				x2.append(ratings[user2][movie])
			x1 = np.array(x1)
			x2 = np.array(x2)
			score = 1/(1+np.sqrt(((x1-x2)**2).sum()))
		scrow.append(score)
	scmat.append(scrow)

print(np.round(scmat, 2))
```

**皮尔逊相关系数（皮氏距离）**

```python
A = [1,2,3,4]
B = [6,5,4,3]
m = np.corrcoef(A, B)  # 相关系数矩阵
```

相关系数 = 协方差/二者标准差之积

处于[-1， 1]区间。越靠近-1，代表越反相关。越靠近1，越正相关，越相似。

```python
"""
demo04_reco.py 推荐引擎
"""
import json
import numpy as np

with open(
	'../ml_data/ratings.json', 'r') as f:
	ratings = json.loads(f.read())

# scmat存储相似度得分矩阵
users, scmat = list(ratings.keys()), []

for user1 in users:
	scrow = []
	for user2 in users:
		movies = set()
		# 把两个人都看过的电影存入movies中
		for movie in ratings[user1]:
			if movie in ratings[user2]:
				movies.add(movie)
		# 二人没有共同语言
		if len(movies) == 0:
			score = 0
		else: 
			# 分别存储两人对同一部电影的打分
			x1, x2 = [], []
			for movie in movies:
				x1.append(ratings[user1][movie])
				x2.append(ratings[user2][movie])
			x1 = np.array(x1)
			x2 = np.array(x2)
			# score = 1/(1+np.sqrt(((x1-x2)**2).sum()))
			score = np.corrcoef(x1, x2)[0,1]
		scrow.append(score)
	scmat.append(scrow)
print(np.round(scmat, 2))

# 按照相似度从高到低排列每个用户的相似用户
scmat = np.array(scmat)
users = np.array(users)
for i, user in enumerate(users):
	# 按照相似度排序 获取降序的索引
	sorted_indices = scmat[i].argsort()[::-1]
	sorted_indices = \
		sorted_indices[sorted_indices!=i]
	# 获取排序后的相似用户
	similar_users = users[sorted_indices]
	# 获取排序后的相似用户的得分
	similar_scores = scmat[i, sorted_indices]
	print(user, similar_users, similar_scores)
```

**生成推荐清单**

1. 找到所有皮尔逊系数正相关的用户。
2. 遍历当前用户的每个相似用户，拿到相似用户看过但是当前用户没有看过的电影作为推荐电影。找个数据结构存一下。

3. 多个相似用户有可能推荐同一部电影，取每个相似用户对该电影的评分的均值作为推荐度。根据该推荐度对电影清单进行排序。

```python
	
	# 找到所有皮尔逊相关系数正相关的用户
	positive_mask = similar_scores > 0 
	similar_users = \
		similar_users[positive_mask]
	# 遍历所有相似用户 整理推荐电影字典
	# dict={'name':[5.0, 3.5, 2.5], ...}
	recomm_movies = {}
	for i, similar_user in enumerate(
			similar_users):
		# 获取相似用户看过但当前用户没看过
		for movie, score in \
			ratings[similar_user].items():
			if movie not in ratings[user].keys():
				if movie not in recomm_movies:
					recomm_movies[movie]=[score]
				else:
					recomm_movies[movie].append(score)

	#对recomm_movies 进行排序
	movie_list=sorted(recomm_movies.items(), 
		key=lambda x: np.average(x[1]), 
		reverse=True)
	print(user)
	print(movie_list)
```



### 自然语言处理(NLP)

Siri工作流程: 1. 听  2. 懂  3.思考  4. 组织语言  5.回答

1. 语音识别
2. 自然语言处理 - 语义分析
3. 业务逻辑分析 - 结合场景 上下文
4. 自然语言处理 - 分析结果生成自然语言文本
5. 语音合成

#### 自然语言处理

自然语言处理的常用处理过程:

先针对训练文本进行分词处理(词干提取, 原型提取), 统计词频, 通过词频-逆文档频率算法获得该词对整个样本语义的贡献, 根据每个词对语义的贡献力度, 构建有监督分类学习模型. 把测试样本交给模型处理, 得到测试样本的语义类别.

自然语言处理工具包 - nltk

#### 文本分词

```python
import nltk.tokenize as tk
# 把一段文本拆分句子
sent_list = tk.sent_tokenize(text)
# 把一句话拆分单词
word_list = tk.word_tokenize(sent)
# 通过文字标点分词器 拆分单词
punctTokenizer=tk.WordPunctTokenizer()
word_list = punctTokenizer.tokenize(text)
```

案例:

```python
"""
demo05_tk.py 文本分词
"""
import nltk.tokenize as tk

doc = "Are you curious about tokenization? " \
	"Let's see how it works! " \
	"We need to analyze a couple of sentences " \
	"with punctuations to see it in action."
print(doc)
# 拆分文档得到句子列表
print('-' * 50)
sents = tk.sent_tokenize(doc)
for i, s in enumerate(sents):
	print(i+1, s)

# 拆分句子得到单词列表
print('-' * 50)
words = tk.word_tokenize(doc)
for i, s in enumerate(words):
	print(i+1, s)

print('-' * 50)
tokens = tk.WordPunctTokenizer()
words = tokens.tokenize(doc)
for i, s in enumerate(words):
	print(i+1, s)
```

#### 词干提取

```python
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb

# 波特词干提取器  (偏宽松)
stemmer = pt.PorterStemmer()
# 朗卡斯特词干提取器   (偏严格)
stemmer = lc.LancasterStemmer()
# 思诺博词干提取器   (偏中庸)
stemmer = sb.SnowballStemmer('english')
r = stemmer.stem('playing') # 词干提取
```

#### 词性还原

与词干提取作用类似, 词干提取出的词干信息不利于人工二次处理(人读不懂), 词性还原可以把名词复数等形式恢复为单数形式. 更有利于人工二次处理.

```python
import nltk.stem as ns
# 词性还原器
lemmatizer = ns.WordNetLemmatizer()
n_lemm=lemmatizer.lemmatize(word, pos='n')
v_lemm=lemmatizer.lemmatize(word, pos='v')
```

案例:

```python
"""
demo07_stem.py 词性还原
"""
import nltk.stem as ns

words = ['table', 'probably', 'wolves', 
	'playing', 'is', 'dog', 'the', 
	'beaches', 'grounded', 'dreamt',
	'envision']

lemmatizer = ns.WordNetLemmatizer()

for word in words:
	n_lemma = lemmatizer.lemmatize(
			word, pos='n')
	v_lemma = lemmatizer.lemmatize(
			word, pos='v')
	print( '%8s %8s %8s' % (
		word, n_lemma, v_lemma) )
```

#### 词袋模型

文本分词处理后, 若需要分析文本语义, 需要把分词得到的结果构建样本模型, 词袋模型就是由每一个句子为一个样本, 单词在句子中出现的次数为特征值构建的数学模型. 

The brown dog is running. The black dog is in the black room. Running in the room is forbidden.

1. The brown dog is running. 
2. The black dog is in the black room.
3. Running in the room is forbidden.

| the  | brown | dog  | is   | running | black | in   | room | forbidden |
| ---- | ----- | ---- | ---- | ------- | ----- | ---- | ---- | --------- |
| 1    | 1     | 1    | 1    | 1       | 0     | 0    | 0    | 0         |
| 2    | 0     | 1    | 1    | 0       | 2     | 1    | 1    | 0         |
| 1    | 0     | 0    | 1    | 1       | 0     | 1    | 1    | 1         |

获取一篇文档的词袋模型:

```python
import sklearn.feature_extraction.text as ft
# 构建词袋模型对象
model = ft.CountVectorizer()
bow = model.fit_transform(sentences)
print(bow)
# 获取词袋模型的特征名
words = model.get_feature_names()
```

案例：

```python
"""
demo08_bow.py  词袋模型
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = "The brown dog is running. " \
	" The black dog is in the black room. "\
	"Running in the black room is forbidden. "
# 拆分句子
sents = tk.sent_tokenize(doc)
print(sents)
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
# 不使用稀疏矩阵输出，用正常矩阵输出
print(bow.toarray())
header = cv.get_feature_names()
print(header)
```

#### 词频(TF)

单词在句子中出现的次数/句子的总词数 称为词频. 即一个单词在句子中出现的频率.  词频相对于单词出现的次数可以更加客观的评估单词对一句话的语义的贡献度. **词频越高,代表当前单词对语义贡献度越大.** 

#### 文档频率(DF)

含有某个单词的文档样本数 / 总文档样本数.

#### 逆文档频率(IDF)

总文档样本数 / 含有某个单词的文档样本数

**单词的逆文档频率越高, 代表当前单词对语义的贡献度越大.**

#### 词频-逆文档频率(TF-IDF)

词频矩阵中的每一个元素乘以相应单词的逆文档频率, 其值越大, 说明该词对样本语义的贡献度越大. 可以根据每个单词的贡献力度, 构建学习模型.

获取TFIDF矩阵相关API:

```python
model = ft.CountVectorizer()
bow = model.fit_transform(sentences)
# 获取IFIDF矩阵
tf = ft.TfidfTransformer()
tfidf = tf.fit_transform(bow)
# 基于tfidf 做模型训练
....
```

案例:

```python
"""
demo09_tfidf.py  词频逆文档频率矩阵
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
import numpy as np

doc = "The brown dog is running. " \
	" The black dog is in the black room. "\
	"Running in the black room is forbidden. "
# 拆分句子
sents = tk.sent_tokenize(doc)
print(sents)
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
# 不使用稀疏矩阵输出，用正常矩阵输出
print(bow.toarray())
header = cv.get_feature_names()
print(header)
# 获取tf-idf样本矩阵
tfidf_model = ft.TfidfTransformer()
tfidf = tfidf_model.fit_transform(bow)
print(np.round(tfidf.toarray(), 1))
```

#### 文本分类(主题识别)

读取20news文件夹, 每个文件夹的文件夹名作为类别标签, 文件夹中的每个文件作为样本, 构建tfidf矩阵, 交给朴素贝叶斯模型训练.   

自定义测试样本, 测试每句话的主题属于哪一类别.

```python
"""
demo10_20news.py 主题识别
"""
import sklearn.datasets as sd
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb
import numpy as np

train = sd.load_files('../ml_data/20news', 
	encoding='latin1', shuffle=True,
	random_state=7)
train_data = np.array(train.data)
train_y = np.array(train.target)
categories = train.target_names
#2000多字符串 每个字符串都是一个文件的内容
print(train_data.shape) 
print(train_y.shape)
print(categories)
# 构建词袋模型bow，根据bow构建tfidf
vectorizer = ft.CountVectorizer()
bow = vectorizer.fit_transform(train_data)
print(bow.shape)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
# 基于多项分布的朴素贝叶斯
model = nb.MultinomialNB()
model.fit(tfidf, train_y)

# 自定义测试样本，主题识别

test_data = [
	"The curveballs of right handed pitchers " \
	" tend to curve to the left ",
	"Caesar cipher is an ancient form of encryption",
	"This two-wheeler is really good on slippery roads"]
# 使用上次的vectorizer对象构建词袋矩阵
test_bow = vectorizer.transform(test_data)
test_tfidf = tt.transform(test_bow)
pred_test_y = model.predict(test_tfidf)
# 把输出的结果以类别名的方式输出
for sent, index in zip(
		test_data, pred_test_y):
	print(sent, '->', categories[index])
```

#### nltk分类器

nltk也提供了朴素贝叶斯分类器，方便的处理NLP的相关分类问题。它可以自动处理词袋模型，完成TFIDF矩阵的整理，最终实现类别预测。

```python
import nltk.classify as cy
import nltk.classify.util as cu
'''
data与sklearn需要的结构不同，如下：
[ ({'age':15, 'score':80},'good'), 
  ({'age':16, 'score':70},'bad'))]
'''
model =  cy.NaiveBayesClassifier.train(data)
# 评估分类器的精准度
ac = cu.accuracy(model. test_data)
print(ac)
```







