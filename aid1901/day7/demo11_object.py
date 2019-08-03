# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo11_object.py 简单物体识别
"""
import os 
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl
import cv2 as cv

def search_files(directory):
    # 读取directory目录的内容，返回一个字典：
    # {'apple':[url1,url2,url3], 'banana':[..]}
    # 把directory目录改为当前平台所能识别的目录
    directory = os.path.normpath(directory)
    objects = {}
    for curdir, subdirs, files in \
        os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                # label -> apple(文件目录名称)
                label = curdir.split(os.path.sep)[-1]
                if label not in objects:
                    objects[label] = []
                path = os.path.join(curdir, file)
                objects[label].append(path)
    return objects

# 整理训练集  为每一个类别训练一个HMM模型
train_samples = search_files(
    '../ml_data/objects/training')

train_x, train_y = [], []
for label, filenames in train_samples.items():
    descs = np.array([])
    for filename in filenames:
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 图像读取后，需要整理图像:伸缩等
        h, w = gray.shape[:2]
        f = 200 / min(w, h)
        # 把gray的宽度与高度按照f比例进行缩放
        gray=cv.resize(gray, None, fx=f, fy=f)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints = sift.detect(gray)
        _,desc = sift.compute(gray, keypoints)
        if len(descs)==0:
            descs = desc
        else:
            descs = np.append(descs, desc, axis=0)
    train_x.append(descs)
    train_y.append(label)

# 基于HMM模型，训练样本
# models: {'apple':<model object>, ...}
models = {}
for mfccs, label in zip(train_x, train_y):
    model = hl.GaussianHMM(n_components=4, 
        covariance_type='diag', 
        n_iter=200)
    models[label] = model.fit(mfccs)

# 针对测试集样本测试
test_samples = search_files(
    '../ml_data/objects/testing')

test_x, test_y = [], []
for label, filenames in test_samples.items():
    descs = np.array([])
    for filename in filenames:
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 图像读取后，需要整理图像:伸缩等
        h, w = gray.shape[:2]
        f = 200 / min(w, h)
        # 把gray的宽度与高度按照f比例进行缩放
        gray=cv.resize(gray, None, fx=f, fy=f)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints = sift.detect(gray)
        _,desc = sift.compute(gray, keypoints)
        if len(descs)==0:
            descs = desc
        else:
            descs = np.append(descs, desc, axis=0)
    test_x.append(descs)
    test_y.append(label)

# test_x与test_y 有7组数据 遍历test_x：
# 使用每个HMM模型对同一个样本进行得分判比
pred_test_y = []
for descs in test_x:
    best_score, best_label = None, None
    for label, model in models.items():
        score = model.score(descs)
        if (best_score is None) or \
           (best_score < score):
            best_score = score
            best_label = label
    pred_test_y.append(best_label)
print(test_y)
print(pred_test_y)









