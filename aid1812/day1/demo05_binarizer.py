"""
demo05_binarizer.py 二值化
"""
import sklearn.preprocessing as sp
import numpy as np

raw_samples = np.array([
	[17, 100, 4000],
	[20, 80, 5000],
	[23, 75, 3500]])

bin = sp.Binarizer(threshold=80)
r = bin.transform(raw_samples)
print(r)