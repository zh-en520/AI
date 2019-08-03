# 机器学习DAY07

#### nltk分类器

nltk也提供了朴素贝叶斯分类器，方便的处理NLP的相关分类问题。它可以自动处理词袋模型，完成TFIDF矩阵的整理，最终实现类别预测。

```python
import nltk.classify as cy
import nltk.classify.util as cu
'''
data与sklearn需要的结构不同，如下：
[ ({'age':15, 'score':80},'good'), 
  ({'age':16, 'score':70},'bad'))]
'''
model =  cy.NaiveBayesClassifier.train(data)
# 评估分类器的精准度
ac = cu.accuracy(model, test_data)
print(ac)
```

#### 情感分析

分析语料库中的movie_reviews文档，通过正面及负面评价进行自然语言模型训练，实现情感分析。

```python
"""
demo01_movie_reviews.py   情感分析
"""
import nltk.corpus as nc
import nltk.classify as nf
import nltk.classify.util as cu
import numpy as np
# 整理正面数据集
# [ ({'age':True, 'score':True},'good'), 
#   ({'email':True, 'pwd':True},'bad'))]

pdata = []
# pos目录下所有文件的路径
fileids = nc.movie_reviews.fileids('pos')
# 把每个文件作为一个样本塞入pdata训练集
for fileid in fileids:
	sample = {}
	# 调用movie_reviews的words方法分词
	words = nc.movie_reviews.words(fileid)
	for word in words:
		sample[word] = True
	pdata.append((sample, 'POSITIVE'))

ndata = []
# pos目录下所有文件的路径
fileids = nc.movie_reviews.fileids('neg')
# 把每个文件作为一个样本塞入pdata训练集
for fileid in fileids:
	sample = {}
	# 调用movie_reviews的words方法分词
	words = nc.movie_reviews.words(fileid)
	for word in words:
		sample[word] = True
	ndata.append((sample, 'NEGATIVE'))

# 拆分测试集与训练集数量  (训练集=0.8)
pnumb, nnumb = \
	int(0.8*len(pdata)), int(0.8*len(ndata))
train_data = pdata[:pnumb]+ndata[:nnumb]
test_data = pdata[pnumb:]+ndata[nnumb:]
# 训练模型
model=nf.NaiveBayesClassifier.train(train_data)
ac = cu.accuracy(model, test_data)
print(ac)

# 模拟业务应用
reviews = [
"It is an amazing movie.", 
"This is a dull movie. I would never recommend it to any one.",
"The cinematography is pretty great in this movie.",
"The direction was terrible and the story was all over the place."]

for review in reviews:
	sample = {}
	words = review.split()
	for word in words:
		sample[word] = True
	pcls = model.classify(sample)
	print(review, '->', pcls)
```

#### 语音识别

解决如何把一个声音识别为相应文本问题。

傅里叶变换可以将时间域的声音信号分解为一系列不同频率的正弦函数的叠加。通过频率谱线的特殊分布，建立音频内容与文本的对应关系，以此作为模型训练基础。

**梅尔频率倒谱系数（MFCC）**

梅尔频率倒谱系数(MFCC)通过与声音内容密切相关的13个特殊频率所对应的能量分布，可以使用MFCC矩阵作为语音识别的特征。最终基于隐马尔科夫模型（HMM）进行模式识别，找到测试样本最匹配的声音模型，从而识别语音内容。

```python
import scipy.io.wavfile as wf
import python_speech_features as sf

sample_rate, sigs = wf.read('f.wav')
mfcc = sf.mfcc(sigs, sample_rate)
```

案例：

```python
"""
demo02_speech.py  mfcc
"""
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf 
import matplotlib.pyplot as mp

sample_rate, sigs = wf.read(
		'../ml_data/speeches/'\
		'training/banana/banana01.wav')
print(sample_rate, sigs.shape)
mfcc = sf.mfcc(sigs, sample_rate)
print('mfcc:', mfcc.shape)

mp.matshow(mfcc.T, cmap='gist_rainbow')
mp.show()
```

##### 隐马尔科夫模型(HMM)

```pip3 install hmmlearn==0.2.1```

```python
import hmmlearn.hmm as hl
# n_components:高斯分布函数的个数
# coveriance_type:使用相关矩阵的辅对角线进
# 行相关性比较
# n_iter:最大迭代上限
model = hl.GaussianHMM(
    n_components=4, 
	covariance_type='diag',
	n_iter=1000)
model.fit(mfccs)
score = model.score(test_mfcc)
```

案例：

```python
"""
demo03_hmm.py  隐马模型实现音频识别
"""
import os
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl


def search_files(directory):
	directory = os.path.normpath(directory)
	# 整理数据
	objects = {}
	# crdir：当前目录
	# subdirs：当前目录下的子目录列表
	# files: 当前目录下的所有文件列表
	for curdir, subdirs, files in \
			os.walk(directory):
		for file in files:
			if file.endswith('.wav'):
				label = curdir.split(os.path.sep)[-1]
				if label not in objects:
					objects[label] = []
				path = os.path.join(curdir, file)
				objects[label].append(path)
	return objects

#读取所有训练数据 
#{'apple':[url,url,url..], 
# 'banana':[url,url,url..] ... }
train_samples = search_files(
	'../ml_data/speeches/training')

# 整理训练集，字典中的每个标签训练一个HMM
train_x, train_y = [], []
for label, filenames in train_samples.items():
	mfccs = np.array([])
	for filename in filenames:
		sample_rate, sigs = wf.read(filename)
		mfcc = sf.mfcc(sigs, sample_rate)
		if len(mfccs) == 0:
			mfccs = mfcc
		else:
			mfccs = np.append(
				mfccs, mfcc, axis=0)
	train_x.append(mfccs)
	train_y.append(label)

# train_x 与 train_y 中包含7组数据
models = {}
for mfccs, label in zip(train_x, train_y):
	model = hl.GaussianHMM(n_components=4, 
		covariance_type='diag', 
		n_iter=1000)
	models[label] = model.fit(mfccs)
# models: {'apple':<HMMmodel>, 'banana':<HMMmodel>}

# 读取测试集中的文件，使用每个HMM模型
# 对测试样本进行评测，取得分最高的模型所在
# 标签作为预测类别。
test_samples = search_files(
	'../ml_data/speeches/testing')
test_x, test_y = [], []
for label, filenames in test_samples.items():
	mfccs = np.array([])
	for filename in filenames:
		sample_rate, sigs = wf.read(filename)
		mfcc = sf.mfcc(sigs, sample_rate)
		if len(mfccs) == 0:
			mfccs = mfcc
		else:
			mfccs = np.append(
				mfccs, mfcc, axis=0)
	test_x.append(mfccs)
	test_y.append(label)

# 遍历测试集样本，使用每个HMM模型对其进行
# 模式匹配评分，取得分最高的为预测类别

pred_test_y = []
for mfccs in test_x:
	best_label, best_score = None, None
	for label, model in models.items():
		score = model.score(mfccs)
		if (best_score is None) or (best_score<score):
			best_label = label
			best_score = score
	# 
	pred_test_y.append(best_label)

print(test_y)
print(pred_test_y)
```



### 图像识别

#### OpenCV基础

OpenCV是一个开源的计算机视觉库。提供了很多图像处理常用的工具。

案例：

```python
"""
demo04_cv.py   opencv基础
"""
import numpy as np
import cv2 as cv

# 读取图片并显示
img = cv.imread('../ml_data/forest.jpg')
print(img.shape)
cv.imshow('Forest.jpg', img)
# 显示图片某个颜色通道的图像
blue = np.zeros_like(img)
blue[:,:,0] = img[:,:,0] #保留了蓝色通道的亮度
cv.imshow('Blue', blue)

green = np.zeros_like(img)
green[:,:,1] = img[:,:,1] #保留了绿色通道的亮度
cv.imshow('Green', green)

red = np.zeros_like(img)
red[:,:,2] = img[:,:,2] #保留了红色通道的亮度
cv.imshow('Red', red)

# 图像裁剪
h, w = img.shape[:2]
l, t = int(w/4), int(h/4)
r, b = int(w*3/4), int(h*3/4)
cropped = img[t:b, l:r]
cv.imshow('Cropped', cropped)

# 图像缩放
s1 = cv.resize(img, (int(w/4), int(h/4)),
	interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled1', s1)

s2 = cv.resize(s1, None, fx=4, fy=4,
	interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled2', s2)

cv.waitKey()  # 阻塞方法

# 图像保存
cv.imwrite('../ml_data/red.jpg', red)
cv.imwrite('../ml_data/s2.jpg', s2)
```

#### 边缘检测

物体的边缘检测是物体识别常用的手段。边缘检测常用亮度梯度方法。通过识别亮度梯度变化最大的像素点从而检测出物体的边缘。

常用边缘检测算法相关API：

```python
# 索贝尔边缘识别
# cv.CV_64F：卷积运算使用数据类型为64位浮点型（保证微分的精度）
# 1：希望在水平方向索贝尔偏微分
# 0：不希望在垂直方向索贝尔偏微分
# ksize：卷积核为5*5的方阵
cv.Sobel(original, cv.CV_64F, 1, 0, ksize=5)
# 拉普拉斯边缘识别
cv.Laplacian(original, cv.CV_64F)
# Canny边缘识别
# 50:水平方向阈值  240:垂直方向阈值
cv.Canny(original, 50, 240)
```

案例：

```python
"""
demo05_canny.py  边缘识别
"""
import cv2 as cv
img = cv.imread('../ml_data/chair.jpg',
	cv.IMREAD_GRAYSCALE)
cv.imshow('Img', img)
# 水平方向索贝尔边缘识别
hs = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('HSobel', hs)
# 垂直方向索贝尔边缘识别
vs = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('VSobel', vs)
s = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Sobel', s)

# 拉普拉斯边缘识别
lap = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Laplacian', lap)

# Canny边缘识别
canny = cv.Canny(img, 50, 200)
cv.imshow('canny', canny)

cv.waitKey()
```



#### 亮度提升

OpenCV提供了直方图均衡化的方式实现亮度提升，更有利于边缘识别与物体识别模型的训练。

OpenCV直方图均衡化相关API：

```python
# 彩色图转为灰度图
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
# 直方图均衡化
equalized_gray = cv.equalizeHist(gray)
```

案例：

```python
"""
demo06_equalizehist.py  直方图均衡化
"""
import cv2 as cv

img = cv.imread('../ml_data/sunrise.jpg')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# 直方图均衡化
equalized_gray = cv.equalizeHist(gray)
cv.imshow('equalized_gray', equalized_gray)
# 对彩色图提高亮度
# YUV：亮度，色度，饱和度
yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
# 单独获取亮度通道，提亮即可
yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0])
color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Color', color)

cv.waitKey()
```

#### 角点检测

平直棱线的交汇点（亮度梯度方向改变的像素点的位置）

OpenCV提供的角点检测相关API：

```python
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
# Harris角点检测器
# 边缘水平方向、垂直方向颜色值改变超过阈值7、5时即为边缘
# 边缘线方向改变超过阈值0.04弧度即为一个角点。
corners = cv.cornerHarris(gray, 7, 5, 0.04)
```

案例：

```python
"""
demo07_corner.py 角点检测
"""
import cv2 as cv

img = cv.imread('../ml_data/box.png')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# 角点检测
corners = cv.cornerHarris(gray, 5, 7, 0.04)
print(corners[corners>0.1])
mixture = img.copy()
mixture[corners>corners.max()*0.01] = (0,0,255)
cv.imshow('Mixture', mixture)

cv.waitKey()
```

#### 特征点检测

常用特征点检测有：STAR特征点检测 / SIFT特征点检测

特征点检测结合了边缘检测与角点检测从而识别出图形的特征点。

STAR特征点检测相关API如下：

```python
import cv2 as cv
# 创建STAR特征点检测器
star = cv.xfeatures2d.StarDetector_create()
# 检测出gray图像所有的特征点
keypoints = star.detect(gray)
# drawKeypoints方法可以把所有的特征点绘制在mixture图像中
cv.drawKeypoints(original, keypoints, mixture,		 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Mixture', mixture)
```

案例：

```python
"""
demo08_star.py 特征点检测器
"""
import cv2 as cv
img = cv.imread('../ml_data/table.jpg')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# star特征点检测器
star = cv.xfeatures2d.StarDetector_create()
keypoints = star.detect(gray)
print(keypoints)
mixture = img.copy()
# 把特征点都输出在mixture图中
cv.drawKeypoints(img, keypoints, 
  mixture, flags= \
  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Mixture', mixture)

cv.waitKey()
```

SIFT特征点检测相关API：

```python
import cv2 as cv

# 创建SIFT特征点检测器
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
```

案例：

```python
"""
demo09_sift.py sift特征点检测器
"""
import cv2 as cv
img = cv.imread('../ml_data/table.jpg')
cv.imshow('Img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
# sift特征点检测器
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
print(keypoints)
mixture = img.copy()
# 把特征点都输出在mixture图中
cv.drawKeypoints(img, keypoints, 
  mixture, flags= \
  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Mixture', mixture)
cv.waitKey()
```

#### 特征值矩阵

图像特征值矩阵（描述）记录了图像的特征点以及每个特征点的梯度信息，相似图像的特征值矩阵也相似。这样只要有足够多的样本，就可以基于隐马尔科夫模型进行图像内容的识别。

特征值矩阵相关API：

```python
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
_, desc = sift.compute(gray, keypoints)
```

案例：

```python
"""
demo10_siftdesc.py sift特征矩阵
"""
import cv2 as cv
import matplotlib.pyplot as mp

img = cv.imread('../ml_data/table.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# sift特征点检测器
sift = cv.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray)
_, desc = sift.compute(gray, keypoints)
print(desc.shape)

mp.matshow(desc.T, cmap='gist_rainbow')
mp.show()
```

#### 物体识别

```python
"""
demo11_objects.py  物体识别 
"""
import os
import numpy as np
import scipy.io.wavfile as wf
import hmmlearn.hmm as hl
import cv2 as cv

def search_files(directory):
	directory = os.path.normpath(directory)
	# 整理数据
	objects = {}
	# crdir：当前目录
	# subdirs：当前目录下的子目录列表
	# files: 当前目录下的所有文件列表
	for curdir, subdirs, files in \
			os.walk(directory):
		for file in files:
			if file.endswith('.jpg'):
				label = curdir.split(os.path.sep)[-1]
				if label not in objects:
					objects[label] = []
				path = os.path.join(curdir, file)
				objects[label].append(path)
	return objects

#读取所有训练数据 
#{'apple':[url,url,url..], 
# 'banana':[url,url,url..] ... }
train_samples = search_files(
	'../ml_data/objects/training')

# 整理训练集，字典中的每个标签训练一个HMM
train_x, train_y = [], []
for label, filenames in train_samples.items():
	descs = np.array([])
	for filename in filenames:
		# 基于cv，加载图片的特征描述矩阵
		image = cv.imread(filename)
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# gray的大小不一致，需要整理一下
		h, w = gray.shape[:2]
		f = 100 / min(h, w)
		gray = cv.resize(gray, None, fx=f, fy=f)
		sift = cv.xfeatures2d.SIFT_create()
		keypoints = sift.detect(gray)
		_, desc = sift.compute(gray, keypoints)

		if len(descs) == 0:
			descs = desc
		else:
			descs = np.append(
				descs, desc, axis=0)
	train_x.append(descs)
	train_y.append(label)

# train_x 与 train_y 中包含3组数据
models = {}
for descs, label in zip(train_x, train_y):
	model = hl.GaussianHMM(n_components=4, 
		covariance_type='diag', 
		n_iter=100)
	models[label] = model.fit(descs)
# models: {'airplne':<HMMmodel>, 'motobike':<HMMmodel>}


# 读取测试集中的文件，使用每个HMM模型
# 对测试样本进行评测，取得分最高的模型所在
# 标签作为预测类别。
test_samples = search_files(
	'../ml_data/objects/testing')
test_x, test_y = [], []
for label, filenames in test_samples.items():
	descs = np.array([])
	for filename in filenames:
		# 基于cv，加载图片的特征描述矩阵
		image = cv.imread(filename)
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# gray的大小不一致，需要整理一下
		h, w = gray.shape[:2]
		f = 100 / min(h, w)
		gray = cv.resize(gray, None, fx=f, fy=f)
		sift = cv.xfeatures2d.SIFT_create()
		keypoints = sift.detect(gray)
		_, desc = sift.compute(gray, keypoints)

		if len(descs) == 0:
			descs = desc
		else:
			descs = np.append(
				descs, desc, axis=0)
	test_x.append(descs)
	test_y.append(label)

# 遍历测试集样本，使用每个HMM模型对其进行
# 模式匹配评分，取得分最高的为预测类别

pred_test_y = []
for descs in test_x:
	best_label, best_score = None, None
	for label, model in models.items():
		score = model.score(descs)
		if (best_score is None) or (best_score<score):
			best_label = label
			best_score = score
	# 
	pred_test_y.append(best_label)

print(test_y)
print(pred_test_y)
```

### 人脸识别

人脸识别与图像识别的区别在于人脸识别需要识别出两个人的不同点。 

#### 视频捕捉

通过OpenCV访问视频捕捉设备（视频头），从而获取图像帧。

视频捕捉相关API：

```python
import cv2 as cv

# 获取视频捕捉设备
video_capture = cv.VideoCapture(0)
# 读取一帧
frame = video_capture.read()[1]
cv.imshow('VideoCapture', frame)
# 释放视频捕捉设备
video_capture.release()
# 销毁cv的所有窗口
cv.destroyAllWindows()
```

案例：

```python
"""
demo12_video.py  捕获视频
"""
import cv2 as cv
# 视频头
video_capture = cv.VideoCapture(0)
while True:
	frame = video_capture.read()[1]
	cv.imshow('Frame', frame)
	if cv.waitKey(33) == 27:
		break

video_capture.release()
cv.destroyAllWindows()
```

#### 人脸定位

哈尔级联人脸定位

```python
import cv2 as cv
# 通过特征描述文件构建哈尔级联人脸识别器
fd = cv.CascadeClassifier('../data/haar/face.xml')
# 从一个图像中识别出所有的人脸区域
# 	1.3：为最小的人脸尺寸
# 	5：最多找5张脸
# 返回：
# 	faces: 抓取人脸（矩形区域）列表 [(l,t,w,h),(),()..]
faces = fd.detectMultiScale(frame, 1.3, 5)
face = faces[0] # 第一张脸
# 绘制椭圆
cv.ellipse(
    face, 				# 图像
    (l + a, t + b), 	# 椭圆心
    (a, b), 			# 半径
    0, 					# 椭圆旋转角度
    0, 360, 			# 起始角, 终止角
    (255, 0, 255), 		# 颜色
    2					# 线宽
)
```

案例：

```python
"""
demo13_cascade.py  级联定位
"""
import cv2 as cv

# 定义哈尔级联人脸定位器
fd = cv.CascadeClassifier('../ml_data/haar/face.xml')
ed = cv.CascadeClassifier('../ml_data/haar/eye.xml')
nd = cv.CascadeClassifier('../ml_data/haar/nose.xml')

# 视频头
video_capture = cv.VideoCapture(0)
while True:
	frame = video_capture.read()[1]
	# 根据哈尔定位器 找到人脸的位置 并绘制
	faces = fd.detectMultiScale(frame, 1.5, 2)
	for l, t, w, h in faces:
		a, b = int(w/2), int(h/2)
		cv.ellipse(frame, 
			(l+a, t+b), 
			(a, b), 
			0, 0, 360, 
			(255,0,255), 2)

		# 找鼻子找眼
		face = frame[t:t+h, l:l+w]
		eyes = ed.detectMultiScale(face, 1.5, 2)
		for l, t, w, h in eyes:
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), 
				(a, b), 
				0, 0, 360, 
				(0,255,255), 2)

		noses = nd.detectMultiScale(face, 1.2, 1)
		for l, t, w, h in noses:
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), 
				(a, b), 
				0, 0, 360, 
				(255,255,0), 2)

	cv.imshow('Frame', frame)
	if cv.waitKey(33) == 27:
		break

video_capture.release()
cv.destroyAllWindows()
```

#### 人脸识别

简单人脸识别：OpenCV的LBPH(局部二值模式直方图)

```python

```















