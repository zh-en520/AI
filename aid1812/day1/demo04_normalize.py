"""
demo04_normalize.py 归一化
"""
import sklearn.preprocessing as sp
import numpy as np

raw_samples = np.array([
	[17, 100, 4000],
	[20, 80, 5000],
	[23, 75, 3500]])

r = sp.normalize(raw_samples, norm='l1')
print(r)
print(abs(r).sum(axis=1))