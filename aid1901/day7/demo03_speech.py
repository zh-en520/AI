# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_speech.py 音频识别
"""
import os 
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl

def search_files(directory):
    # 读取directory目录的内容，返回一个字典：
    # {'apple':[url1,url2,url3], 'banana':[..]}
    # 把directory目录改为当前平台所能识别的目录
    directory = os.path.normpath(directory)
    objects = {}
    for curdir, subdirs, files in \
        os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                # label -> apple(文件目录名称)
                label = curdir.split(os.path.sep)[-1]
                if label not in objects:
                    objects[label] = []
                path = os.path.join(curdir, file)
                objects[label].append(path)
    return objects

# 整理训练集  为每一个类别训练一个HMM模型
train_samples = search_files(
    '../ml_data/speeches/training')

train_x, train_y = [], []
for label, filenames in train_samples.items():
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs)==0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    train_x.append(mfccs)
    train_y.append(label)

# 基于HMM模型，训练样本
# models: {'apple':<model object>, ...}
models = {}
for mfccs, label in zip(train_x, train_y):
    model = hl.GaussianHMM(n_components=4, 
        covariance_type='diag', 
        n_iter=1000)
    models[label] = model.fit(mfccs)

# 针对测试集样本测试
test_samples = search_files(
    '../ml_data/speeches/testing')

test_x, test_y = [], []
for label, filenames in test_samples.items():
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs)==0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    test_x.append(mfccs)
    test_y.append(label)

# test_x与test_y 有7组数据 遍历test_x：
# 使用每个HMM模型对同一个样本进行得分判比
pred_test_y = []
for mfccs in test_x:
    best_score, best_label = None, None
    for label, model in models.items():
        score = model.score(mfccs)
        if (best_score is None) or \
           (best_score < score):
            best_score = score
            best_label = label
    pred_test_y.append(best_label)
print(test_y)
print(pred_test_y)









