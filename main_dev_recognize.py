'''
SUMMARY:  Load trained model and Evaluate on test dataset
Usage:    * fold, fe_fd, agg_num should be same as train procedure
          * model to load should be specified
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: 2016.10.09 Modify varibale names
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from hat.optimizers import Rmsprop
from hat import serializations
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data
import csv
import cPickle
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)

# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11    # should be same as training phase
hop = 5         # should be same as training phase
fold = 0        # should be same as training phase
n_labels = len( cfg.labels )

# load model
md = serializations.load( cfg.dev_md+'/md10.p' )

# get scaler
scaler = pp_dev_data.Scaler( fe_fd, cfg.dev_tr_csv[fold] )

# do recognize and evaluation
n_labels = len( cfg.labels )
confM = np.zeros( (n_labels, n_labels) )      # confusion matrix
acc_frs = []

# get test file names
with open( cfg.dev_te_csv[fold], 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)
    
# predict for each scene
for li in lis:
    # load data
    [na, lb] = li[0].split('\t')
    na = na.split('/')[1][0:-4]
    path = fe_fd + '/' + na + '.f'
    X = cPickle.load( open( path, 'rb' ) )
    X = scaler.transform( X )
    X = mat_2d_to_3d( X, agg_num, hop )

    # predict
    p_y_preds = md.predict(X)        # probability, size: (n_block,label)
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    pred = int( b[0] )
    id = cfg.lb_to_id[lb]
    
    confM[ id, pred ] += 1            
    corr_fr = list(preds).count(id)     # correct frames
    acc_frs += [ float(corr_fr) / X.shape[0] ]   # frame accuracy
        
acc = np.sum( np.diag( np.diag( confM ) ) ) / np.sum( confM )
acc_fr = np.mean( acc_frs )

print 'event_acc:', acc
print 'frame_acc:', acc_fr
print confM