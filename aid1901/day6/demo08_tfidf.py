# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_tfidf.py  词袋模型
"""
import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
import numpy as np
doc = "The brown dog is running. " \
	  "The black dog is in the black room. " \
	  "Running in the room is forbidden. "
sents = tk.sent_tokenize(doc)
# 构建词袋模型
cv = ft.CountVectorizer()
bow = cv.fit_transform(sents)
print(bow.toarray())
print(cv.get_feature_names())

# 获取TFIDF矩阵
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
print(np.round(tfidf.toarray(), 2))
