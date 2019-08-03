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


