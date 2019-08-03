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


