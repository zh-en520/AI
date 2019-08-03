"""
demo02_scale.py 均值移除
"""
import sklearn.preprocessing as sp
import numpy as np

raw_samples = np.array([
	[17, 100, 4000],
	[20, 80, 5000],
	[23, 75, 3500]])
A = sp.scale(raw_samples)
print(A)
print(A.mean(axis=0))  #每列的均值
print(A.std(axis=0))  #每列的标准差