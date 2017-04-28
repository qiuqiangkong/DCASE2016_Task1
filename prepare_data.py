"""
SUMMARY:  prepareData, some functions copy from py_sceneClassification2
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.10.09 modify variable name
--------------------------------------
"""
import csv
import wavio
import sys
import numpy as np
from scipy import signal
import scipy.stats
import cPickle
import os
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import config as cfg
sys.path.append(cfg.hat_root)
from hat.preprocessing import mat_2d_to_3d, sparse_to_categorical

def readwav(path):
    Struct = wavio.read(path)
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

def to_mono(wav):
    if wav.ndim == 1:
        return wav
    elif wav.ndim == 2:
        return np.mean(wav, axis=-1)

def calculate_logmel(wav_fd, fe_fd):
    """Calculate log mel spectrogram and write to disk. 
    
    Args:
      wav_fd: string. 
      fe_fd: string. Calculated features will be saved here. 
    """
    names = [na for na in os.listdir(wav_fd) if na.endswith('.wav')]
    names = sorted(names)
    for na in names:
        print na
        path = os.path.join(wav_fd, na)
        wav, fs = readwav(path)
        wav = to_mono(wav)
        assert fs == cfg.fs
        ham_win = np.hamming(cfg.n_fft)
        [f, t, x] = signal.spectral.spectrogram(x=wav, 
                                                window=ham_win, 
                                                nperseg=cfg.n_fft, 
                                                noverlap=0, 
                                                detrend=False, 
                                                return_onesided=True, 
                                                mode='magnitude') 
        x = x.T     # (n_frames, n_freq)
        
        # Mel transform matrix
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel(sr=fs, 
                                       n_fft=cfg.n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
            #melW /= np.max(melW, axis=-1)[:,None]
            
        x = np.dot(x, melW.T)
        x = np.log(x + 1e-8)
        
        # # DEBUG. print mel-spectrogram
        # plt.matshow(x.T, origin='lower', aspect='auto')
        # plt.show()
        # pause
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def get_scaler(fe_fd, csv_file, with_mean, with_std):
    """Calculate scaler from data in csv_file. 
    
    Args:
      fe_fd: string. Feature folder. 
      csv_file: string. Path of csv file. 
      with_mean: bool. 
      with_std: bool. 
      
    Returns:
      scaler object. 
    """
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    x_all = []
    for li in lis:
        try:
            [na, lb] = li[0].split('\t')
        except:
            na = li[0]
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        x_all.append(x)
    
    x_all = np.concatenate(x_all, axis=0)
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler(with_mean, with_std).fit(x_all)
    return scaler
    
def get_matrix_format_data(fe_fd, csv_file, n_concat, hop, scaler):
    """Get training data and ground truth in matrix format. 
    
    Args:
      fe_fd: string. Feature folder. 
      csv_file: string. Path of csv file. 
      n_concat: integar. Number of frames to concatenate. 
      hop: integar. Number of hop frames. 
      scaler: None | object. 
    """
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    x3d_all = []
    y_all = []
    
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        if scaler:
            x = scaler.transform(x)
        x3d = mat_2d_to_3d(x, n_concat, hop)     # (n_blocks, n_concat, n_freq)
        x3d_all.append(x3d)
        y_all += [cfg.lb_to_id[lb]] * len(x3d)
    
    x3d_all = np.concatenate(x3d_all)       # (n_samples, n_concat, n_freq)
    y_all = np.array(y_all)            
    y_all = sparse_to_categorical(y_all, len(cfg.labels)) # (n_samples, n_labels)
    return x3d_all, y_all

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def get_mode_value(ary):
    return scipy.stats.mode(ary)[0]
        
if __name__ == "__main__":
    create_folder(cfg.dev_fe_logmel_fd)
    create_folder(cfg.eva_fe_logmel_fd)
    
    # calculate mel feature
    calculate_logmel(cfg.dev_wav_fd, cfg.dev_fe_logmel_fd)
    calculate_logmel(cfg.eva_wav_fd, cfg.eva_fe_logmel_fd)