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
	print(objects)
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