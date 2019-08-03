# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_nltk_cls.py nltk分类器  情感分类
"""

import nltk.corpus as nc
import nltk.classify as cf
import nltk.classify.util as cu
pdata = []

# pos文件夹中的每个文件的路径
fileids = nc.movie_reviews.fileids('pos')
# 整理所有正面评论单词，存入pdata列表
for fileid in fileids:
    sample = {}
    # 分词
    words = nc.movie_reviews.words(fileid)
    for word in words:
        sample[word] = True
    pdata.append((sample, 'POSITIVE'))
# 整理所有正面评论单词，存入ndata列表
ndata = []
fileids = nc.movie_reviews.fileids('neg')
for fileid in fileids:
    sample = {}
    words = nc.movie_reviews.words(fileid)
    for word in words:
        sample[word] = True
    ndata.append((sample, 'NEGATIVE'))

# 拆分测试集与训练集数量（80%作为训练集）
pnumb, nnumb = int(0.8 * len(pdata)), int(0.8 * len(ndata))
train_data = pdata[:pnumb] + ndata[:nnumb]
test_data = pdata[pnumb:] + ndata[nnumb:]
# 基于朴素贝叶斯分类器训练测试数据 
model = cf.NaiveBayesClassifier.train(train_data)
ac = cu.accuracy(model, test_data)
print(ac)

# 模拟业务场景
reviews = [
    'It is an amazing movie.',
    'This is a dull movie. I would never recommend it to anyone.',
    'The cinematography is pretty great in this movie.',
    'The direction was terrible and the story was all over the place.']
sents, probs = [], []
for review in reviews:
    sample = {}
    words = review.split()
    for word in words:
        sample[word] = True
    pcls = model.classify(sample)
    print(review, '->', pcls)