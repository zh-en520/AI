# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_mfcc.py  mfcc矩阵
"""
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp

sample_rate, sigs = \
    wf.read('../ml_data/speeches/training'\
            '/banana/banana02.wav')
print(sample_rate, sigs.shape)
#获取mfcc矩阵
mfcc = sf.mfcc(sigs, sample_rate)
print(mfcc.shape)

mp.matshow(mfcc.T, cmap='gist_rainbow')
mp.show()









