"""
demo14_lbph.py  局部模式直方图人脸识别
"""
import os
import numpy as np
import cv2 as cv
import sklearn.preprocessing as sp

fd = cv.CascadeClassifier('../ml_data/haar/face.xml')

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

train_faces = search_files(
	'../ml_data/faces/training')
codec = sp.LabelEncoder()
codec.fit(list(train_faces.keys()))

train_x, train_y = [], []
for label, filenames in train_faces.items():
	for filename in filenames:
		image = cv.imread(filename)
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		faces = fd.detectMultiScale(
			gray, 1.2, 1, minSize=(100,100))
		for l, t, w, h in faces:
			train_x.append(gray[t:t+h, l:l+w])
			train_y.append(
				codec.transform([label])[0])
train_y = np.array(train_y)

# 构建LBPH模型 识别人脸
model = cv.face.LBPHFaceRecognizer_create()
model.train(train_x, train_y)

# 测试
test_faces = search_files(
	'../ml_data/faces/testing')
test_x, test_y = [], []
for label, filenames in test_faces.items():
	print('outer1 for loop .........')
	for filename in filenames:
		print('outer2 for loop .........')
		image = cv.imread(filename)
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		faces = fd.detectMultiScale(
			gray, 1.2, 1, minSize=(100,100))
		for l, t, w, h in faces:
			print('add face ...')
			test_x.append(gray[t:t+h, l:l+w])
			test_y.append(
				codec.transform([label])[0])

test_y = np.array(test_y)

# 预测 
pred_test_y = []
for face in test_x:
	pred_code = model.predict(face)[0]
	pred_test_y.append(pred_code)

# 验证结果
print(codec.inverse_transform(test_y))
print(codec.inverse_transform(pred_test_y))
