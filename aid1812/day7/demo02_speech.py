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


