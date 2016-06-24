'''
SUMMARY:  prepareData, some functions copy from py_sceneClassification2
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
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
import co_occurrence as co
from Hat.preprocessing import mat_2d_to_3d
import stats

### calculate features
# extract mel feature
# Use preemphasis, the same as matlab
def GetMel( wav_fd, fe_fd, n_delete ):
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
            melW = librosa.filters.mel( fs, n_fft=1024, n_mels=40, fmin=0., fmax=22100 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        
        # DEBUG. print mel-spectrogram
        #plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        
def GetBankSpectrogram( wav_fd, fe_fd, banks=None, n_delete=0 ):
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
        X = X.T		# size: N*(nperseg/2+1)

        if banks is not None:
            X = np.dot( X, banks )
        
        X = X[:, n_delete:]
        #plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        #plt.show()
        #pause 
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        
        
###


def GetBlockStat( X ):
    X = np.log( X )     # map to log scale

    # get distribution of data
    edges = np.arange(-15,5,0.3)
    hist, _ = np.histogram( X, edges )
    hist = hist.astype( float )
    p = hist / np.sum(hist)
    
    # get stats of distribution
    mean_ = stats.mean( p )
    var_ = stats.variance( p )
    skew_ = stats.skewness( p )
    kurt_ = stats.kurtosis( p )
    energy_ = stats.energy( p )
    entropy_ = stats.entropy( p )
    
    sts = [ mean_, var_, skew_, kurt_, energy_, entropy_ ]
    return sts

def GetCoMatrixStats( P ):
    energy_ = co.energy( P )
    mean_x_ = co.mean( P, 'row' )
    mean_y_ = co.mean( P, 'col' )
    var_x_ = co.variance( P, 'row' )
    var_y_ = co.variance( P, 'col' )
    correlation_ = co.correlation( P )
    inertia_ = co.inertia( P )
    absolute_value_ = co.absolute_value( P )
    inverse_difference_ = co.inverse_difference( P )
    entropy_ = co.entropy( P )
    
    return [ energy_, mean_x_, mean_y_, var_x_, var_y_, correlation_, inertia_, absolute_value_, inverse_difference_,  entropy_ ]

def GetBlockStat2( X ):
    X = np.log( X )     # map to log scale

    # get distribution of data
    bgn, fin, interval = -15, 5, 0.5
    sts = []
    
    M, P = co.co_occurrence_matrix( X, bgn, fin, interval, theta='0' )
    sts += GetCoMatrixStats( P )
    
    M, P = co.co_occurrence_matrix( X, bgn, fin, interval, theta='90' )
    sts += GetCoMatrixStats( P )
    
    return sts


# eg. size of input X is 10*513
def GetSliceStat( X, order ):
    edges = [ [0, 100], [50, 150], [100, 200], [150, 250], [200, 300], [250, 350], [300, 400], [350, 450], [400, 500]  ]
    sts = []
    for n in xrange(len(edges)):
        block = X[ :, edges[n][0]:edges[n][1] ]
        if order==1:
            sts += GetBlockStat( block )
        if order==2:
            sts += GetBlockStat2( block )
    return sts
    
        
def GetSpectrogramStat( sp_fe_fd, out_fe_fd, agg_num, hop, order ):
    n_block = 3
    names = [ na for na in os.listdir(sp_fe_fd) ]
    names = sorted(names)
    cnt = 0
    for na in names:
        print cnt, na
        X = cPickle.load( open( sp_fe_fd+'/'+na, 'rb' ) )
        (N, n_in) = X.shape

        pt = 0
        Sts = []
        while (pt + agg_num) < N:
            Xslice = X[pt:pt+agg_num]
            sts = GetSliceStat( Xslice, order )
            Sts.append( sts )
            pt += hop
        Sts = np.array( Sts )
        
        out_path = out_fe_fd + '/' + na[0:-2] + '.f'
        cPickle.dump( Sts, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1

###

def GetDictData0( fe_fd, csv_file ):
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
        
        # insert to Dict
        Dict[lb].append( X )
    
    return Dict

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
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        # insert to Dict
        Dict[lb].append( X3d )
    
    return Dict
   
# concatenate all dict data to mat data
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

# size(Xall): (n_sample, n_time, n_freq)
# size(yall): (n_sample)
def LoadDataToMat( fe_fd, csv_file ):
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xall = []
    yall = []
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        Xall.append( X )
        yall += [ cfg.lb_to_id[lb] ] * len(X) 
        
    Xall = np.concatenate( Xall, axis=0 )
    yall = np.array( yall )
    return Xall, yall
    
def LoadDataToMat2( fe_fd, csv_file, agg_num, hop ):
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xall = []
    yall = []
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        X3d = mat_2d_to_3d( X, agg_num, hop )
        Xall.append( X3d )
        yall += [ cfg.lb_to_id[lb] ] * len(X3d) 
        
    Xall = np.concatenate( Xall, axis=0 )
    yall = np.array( yall )
    return Xall, yall
    
# size(Xall): (n_song, n_seg, n_time, n_freq)
# size(yall): (n_song, n_seg)
def Load4dData( fe_fd, csv_file ):
    # read csv
    with open( csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xall = []
    yall = []
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        Xall.append( X )
        yall.append( [ cfg.lb_to_id[lb] ] * len(X) )
        
    Xall = np.array( Xall )
    yall = np.array( yall )
    return Xall, yall

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
    
def ImageToLevel( X3d ):
    Level3d = np.log( X3d )
    #level = list( np.arange(-4, 15) )
    Level3d = np.clip( Level3d, -7.9, 1.9 )
    Level3d *= 2
    Level3d = Level3d.astype(int)
    return Level3d
    
# texture_type: '0' | '90'
def get_3d_feature( agg_num, hop, texture_type='0' ):
    # get 3d mel feature
    fe_fd = cfg.fe_mel_fd
    names =  os.listdir( fe_fd )
    names = sorted(names)
    cnt = 0
    for na in names:
        print cnt
        path = fe_fd + '/' + na[0:-2] + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        
        # get 3d mel feature and dump
        X3d = mat_2d_to_3d( X, agg_num, hop )
        outpath = cfg.fe_mel3d_fd + '/' + na[0:-2] + '.f'
        cPickle.dump( X3d, open(outpath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1

    # get 3d texture feature
    names = os.listdir( cfg.fe_fft_fd )
    names = sorted( names )
    cnt = 0
    for na in names:
        print cnt 
        path = cfg.fe_fft_fd + '/' + na[0:-2] + '.f'
        X = cPickle.load( open( path, 'rb' ) )
        
        X3d = mat_2d_to_3d( X, agg_num, hop )
        level_list = list( np.arange(-16,4) )
        levels = { ch:i for i, ch in enumerate(level_list) }
        Level3d = ImageToLevel( X3d )

        # texture feature
        Zs = []
        for n in xrange( len(X3d) ):
            # todo 0, 90
            M, P = co_occurrence_matrix( Level3d[n,:,:], levels, texture_type )
            Zs.append( P.flatten() )
        Zs = np.array( Zs )
        if texture_type=='0':
            outpath = cfg.fe_texture3d0_fd + '/' + na[0:-2] + '.f'
        if texture_type=='90':
            outpath = cfg.fe_texture3d90_fd + '/' + na[0:-2] + '.f'
        cPickle.dump( Zs, open(outpath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1
        
    
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
if __name__ == "__main__":
    CreateFolder('Fe')
    CreateFolder('Fe/Mel')
    CreateFolder('Fe/Fft')
    CreateFolder('Fe/SpStat')
    CreateFolder('Fe/SpStat2')
    CreateFolder('Fe/Mel3d')
    CreateFolder('Fe/Fft3d')
    CreateFolder('Fe/Texture3d0')
    CreateFolder('Fe/Texture3d90')
    CreateFolder('Results')
    CreateFolder('Md')
    
    # calculate mel feature
    GetMel( cfg.wav_fd, cfg.fe_mel_fd, n_delete=0 )
    
    # calculate bank feature
    #GetBankSpectrogram( cfg.wav_fd, cfg.fe_fft_fd, banks=None, n_delete=0 )
    #GetSpectrogramStat( cfg.fe_fft_fd, cfg.fe_sp_stat_fd, agg_num=10, hop=10, order=1 )
    #GetSpectrogramStat( cfg.fe_fft_fd, cfg.fe_sp_stat2_fd, agg_num=10, hop=10, order=2 )

    '''
    # calculate texture feature
    agg_num = 10
    hop = 10
    get_3d_feature( agg_num, hop, texture_type='0' )
    get_3d_feature( agg_num, hop, texture_type='90' )
    '''