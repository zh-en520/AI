"""
demo06_onehot.py  独热编码
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[1, 3, 2], 
						[7, 5, 4], 
						[1, 8, 6], 
						[2, 3, 9]])
# 创建独热编码器
ohe=sp.OneHotEncoder(sparse=False,dtype=int)
# fit方法意味着：训练后得到编码码表
ohe_dict = ohe.fit(raw_samples)
print(ohe_dict)
r = ohe_dict.transform(raw_samples)
print(r)


