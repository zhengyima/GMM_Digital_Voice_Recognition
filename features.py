import wave
import os
import numpy as np
from struct import unpack
import pyaudio
import scipy.io.wavfile as wav
from python_speech_features import *
from tqdm import tqdm

import argparse

def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, samplerate=fs,numcep=13, winlen=0.025, winstep=0.01,nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
    

    d_mfcc_feat = delta(mfcc_feature, 1)
    d_mfcc_feat2 = delta(mfcc_feature, 2)
    # test = delta(mfcc_feature, 1)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    
    
    # mfcc_feature = np.hstack((mfcc_feature, d_mfcc_feat, d_mfcc_feat2))
    # mfcc_feature = np.hstack((mfcc_feature))
    # print(mfcc_feature.shape)
    return mfcc_feature
