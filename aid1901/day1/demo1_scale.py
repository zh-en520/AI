import numpy as np
import sklearn.preprocessing as sp

samples = np.array([
    [17.,100.,4000.],
    [20.,80.,5000.],
    [23.,75.,5500.]
])
#均值移除(每一列均值为０，标准差为１)
std_samples = sp.scale(samples)
print(std_samples)
print(std_samples.mean(axis=0))
print(std_samples.std(axis=0))

#范围缩放
mms = sp.MinMaxScaler(feature_range=(0,1))
result = mms.fit_transform(samples)
print(result)

#手动实现范围缩放
for col in samples.T:
    #17K+b=0
    #23K+b=1
    A = np.array([[col.min(),1],
                  [col.max(),1]])
    B = np.array([0,1])
    kb = np.linalg.solve(A,B)
    print(col*kb[0]+kb[1])

#归一化
r = sp.normalize(samples,norm='l2')
print(r)

# #二值化　阈值
# bin = sp.Binarizer(threshold=80)
# result = bin.transform(samples)
# print(result)