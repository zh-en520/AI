"""
demo01_movie_reviews.py   情感分析
"""
import nltk.corpus as nc
import nltk.classify as nf
import nltk.classify.util as cu
import numpy as np
# 整理正面数据集
# [ ({'age':True, 'score':True},'good'), 
#   ({'email':True, 'pwd':True},'bad'))]

pdata = []
# pos目录下所有文件的路径
fileids = nc.movie_reviews.fileids('pos')
# 把每个文件作为一个样本塞入pdata训练集
for fileid in fileids:
	sample = {}
	# 调用movie_reviews的words方法分词
	words = nc.movie_reviews.words(fileid)
	for word in words:
		sample[word] = True
	pdata.append((sample, 'POSITIVE'))

ndata = []
# pos目录下所有文件的路径
fileids = nc.movie_reviews.fileids('neg')
# 把每个文件作为一个样本塞入pdata训练集
for fileid in fileids:
	sample = {}
	# 调用movie_reviews的words方法分词
	words = nc.movie_reviews.words(fileid)
	for word in words:
		sample[word] = True
	ndata.append((sample, 'NEGATIVE'))

# 拆分测试集与训练集数量  (训练集=0.8)
pnumb, nnumb = \
	int(0.8*len(pdata)), int(0.8*len(ndata))
train_data = pdata[:pnumb]+ndata[:nnumb]
test_data = pdata[pnumb:]+ndata[nnumb:]
# 训练模型
model=nf.NaiveBayesClassifier.train(train_data)
ac = cu.accuracy(model, test_data)
print(ac)

# 模拟业务应用
reviews = [
"It is an amazing movie.", 
"This is a dull movie. I would never recommend it to any one.",
"The cinematography is pretty great in this movie.",
"The direction was terrible and the story was all over the place."]

for review in reviews:
	sample = {}
	words = review.split()
	for word in words:
		sample[word] = True
	pcls = model.classify(sample)
	print(review, '->', pcls)