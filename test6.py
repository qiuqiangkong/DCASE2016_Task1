# combine mel, texture, do DNN
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
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
import csv
import cPickle
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)

# hyper-params

agg_num = 10
hop = 10
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md/md500.p', 'rb' ) )

# do recognize and evaluation
n_labels = len( cfg.labels )
confM = np.zeros( (n_labels, n_labels) )      # confusion matrix
acc_frs = []

# get test file names
with open( cfg.te_csv[0], 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)
    
for li in lis:
    [na, lb] = li[0].split('\t')
    na = na.split('/')[1][0:-4]
    mel_path = cfg.fe_mel3d_fd + '/' + na + '.f'
    texture0_path = cfg.fe_texture3d0_fd + '/' + na + '.f'
    texture90_path = cfg.fe_texture3d90_fd + '/' + na + '.f'
    Xmel = cPickle.load( open( mel_path, 'rb' ) )
    Xtexture0 = cPickle.load( open( texture0_path, 'rb' ) )
    Xtexture90 = cPickle.load( open( texture90_path, 'rb' ) )

    # predict
    p_y_preds = md.predict([Xmel, Xtexture0, Xtexture90])        # probability, size: (n_block,label)
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    pred = int( b[0] )
    id = cfg.lb_to_id[lb]
    confM[ id, pred ] += 1            
    corr_fr = list(preds).count(id)     # correct frames
    acc_frs += [ float(corr_fr) / Xmel.shape[0] ]   # frame accuracy
        
acc = np.sum( np.diag( np.diag( confM ) ) ) / np.sum( confM )
acc_fr = np.mean( acc_frs )

print 'event_acc:', acc
print 'frame_acc:', acc_fr
print confM