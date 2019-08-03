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



