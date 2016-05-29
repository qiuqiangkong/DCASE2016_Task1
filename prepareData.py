'''
SUMMARY:  prepareData, some functions copy from py_sceneClassification2
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
import numpy as np
from scipy import signal
import cPickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scikits.audiolab import wavread
import librosa
import config as cfg
import csv
import scipy.stats

# extract mel feature
# Use preemphasis, the same as matlab
def GetMel( wav_fd, fe_fd ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs, enc = wavread( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==44100
        ham_win = np.hamming(1024)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=1024, n_mels=60, fmin=0., fmax=22100 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        
        # DEBUG. print mel-spectrogram
        #plt.matshow(X.T, origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        
# Read all features to Dict    
# the key of Dict is label, value is list of 3d (n_block, n_time, n_freq) audio features
def GetDictData( fe_fd, csv_file, agg_num, hop ):
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # init Dict
    Dict = {}
    for e in cfg.labels:
        Dict[e] = []
        
    # Add feature to Dict
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        
        # reshape data to (n_block, n_time, n_freq)
        [len_X, n_freq] = X.shape
        X3d = []    # shape: (n_block, n_time, n_freq)
        i1 = 0
        while ( i1+agg_num<len_X ):
            X3d.append( X[i1:i1+agg_num] )
            i1 += hop
        X3d = np.array( X3d )
        
        # insert to Dict
        Dict[lb].append( X3d )
    
    return Dict
   
# format dict data to mat data
# size(X): (batch_num, n_time, n_freq)
# size(y): (batch_num, n_labels)
def DictToMat( Dict ):
    Xlist = []
    ylist = []      # 1 of 15 acoustic label
    y2list = []     # 1 of 3 scene label
    
    for ke in Dict.keys():
        for X in Dict[ke]:
            scene = cfg.acoustic_to_scene[ke]
            n_block = X.shape[0]
            Xlist.append( X )
            ylist.append( cfg.lb_to_id[ke] * np.ones(n_block) )
            y2list.append( cfg.lb2_to_id[ scene ] * np.ones(n_block) )
            
    Xall = np.concatenate( Xlist, axis=0 )
    yall = np.concatenate( ylist, axis=0 )
    y2all = np.concatenate( y2list, axis=0 )
    return Xall, yall, y2all


# Recognize all files in teDict
def Recognize( f_predict, teDict, n_out, scaler=None ):
    confM = np.zeros( (n_out, n_out) )      # confusion matrix
    acc_frs = []
    for ke in teDict.keys():
        for X in teDict[ke]:
            if scaler is not None:
                X = scaler.transform(X)
            p_y_preds = f_predict(X)        # preds
            preds = np.argmax( p_y_preds, axis=-1 )
            b = scipy.stats.mode(preds)
            pred = int( b[0] )
            id = cfg.lb_to_id[ke]
            confM[ id, pred ] += 1            
            corr_fr = list(preds).count(id)     # correct frames
            acc_frs += [ float(corr_fr) / X.shape[0] ]   # frame accuracy
            
    acc = np.sum( np.diag( np.diag( confM ) ) ) / np.sum( confM )
    acc_fr = np.mean( acc_frs )
    return acc, acc_fr, confM
        
if __name__ == "__main__":
    GetMel( cfg.wav_fd, cfg.fe_mel_fd )