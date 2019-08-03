import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [1,3,2],
    [7,5,4],
    [1,8,6],
    [7,3,9]
])
ohe = sp.OneHotEncoder(sparse=True,dtype='int32')#sparse紧缩格式,sparse=False稀疏矩阵
r = ohe.fit_transform(samples)
print(r)

#标签编码
features = np.array(['bmw','ford','toyota','audi','auto','ford','toyota','wlhg','hongqi'])
lbe = sp.LabelEncoder()
r = lbe.fit_transform(features)
print(r)
features = lbe.inverse_transform(r)
print(features)