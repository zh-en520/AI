"""
demo07_lbe.py 标签编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples=np.array(['audi', 'ford',
	'audi', 'toyota', 'ford', 'bmw',
	'toyota', 'ford', 'audi'])

# 执行标签编码
lbe = sp.LabelEncoder()
r = lbe.fit_transform(raw_samples)
print(r)
# 通过数字向量获取对应的字符串向量
ir = lbe.inverse_transform([0,1,1])
print(ir)
