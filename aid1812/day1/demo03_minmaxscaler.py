"""
demo03_minmaxscaler.py 范围缩放
"""
import sklearn.preprocessing as sp
import numpy as np

raw_samples = np.array([
	[17, 100, 4000],
	[20, 80, 5000],
	[23, 75, 3500]])

mms = sp.MinMaxScaler(feature_range=(0,1))
r = mms.fit_transform(raw_samples)
print(r)