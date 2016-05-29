'''
SUMMARY:  Load trained model and Evaluate on test dataset
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: -
--------------------------------------
'''
import sys
sys.path.append('../Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10
hop = 10
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md/md10.p', 'rb' ) )

# load test data
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[0], agg_num, hop )

# do recognize and evaluation
n_labels = len( cfg.labels )
confM = np.zeros( (n_labels, n_labels) )      # confusion matrix
acc_frs = []
for ke in teDict.keys():
    for X in teDict[ke]:
        p_y_preds = md.predict(X)        # probability, size: (n_block,label)
        preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
        b = scipy.stats.mode(preds)
        pred = int( b[0] )
        id = cfg.lb_to_id[ke]
        confM[ id, pred ] += 1            
        corr_fr = list(preds).count(id)     # correct frames
        acc_frs += [ float(corr_fr) / X.shape[0] ]   # frame accuracy
        
acc = np.sum( np.diag( np.diag( confM ) ) ) / np.sum( confM )
acc_fr = np.mean( acc_frs )

print 'event_acc:', acc
print 'frame_acc:', acc_fr
print confM