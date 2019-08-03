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