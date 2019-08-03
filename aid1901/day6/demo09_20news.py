# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_20news.py
"""
import sklearn.datasets as sd
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb
import numpy as np

train = sd.load_files('../ml_data/20news', 
	encoding='latin1', shuffle=True, 
	random_state=7)
train_data = train.data
train_y = train.target
categories = train.target_names
print(np.array(train_data).shape)
print(np.array(train_y).shape)
print(categories)

# 构建TFIDF矩阵
cv = ft.CountVectorizer()
bow = cv.fit_transform(train_data)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
# 模型训练  使用MultinomialNB 是因为tfidf
# 矩阵中样本的分布更匹配多项分布
model = nb.MultinomialNB()
model.fit(tfidf, train_y)

# 测试
test_data = [
	'The curveballs of right handed pithers '\
	'tend to curve to the left.',
	'Caesar cipher is an ancient form of encryption.',
	'This two-wheeler is really good on slippery roads.'
]
test_bow = cv.transform(test_data)
test_tfidf = tt.transform(test_bow)
pred_test_y = model.predict(test_tfidf)

for sent, index in zip(test_data, pred_test_y):
	print(sent, '->', categories[index])














