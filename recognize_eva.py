'''
SUMMARY:  Do classification on private dataset and write out result. 
          Training time: 9 s/epoch. (Tesla M2090)
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from Hat.optimizers import Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData
import csv
import cPickle

# hyper-params
fe_fd = cfg.fe_mel_eva_fd
agg_num = 10    # this should be same as training procedure
hop = 10
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md_eva/md100.p', 'rb' ) )

# load name of wavs to be classified
with open( cfg.txt_eva_path, 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)

# do classification for each file
names = []
pred_lbs = []

for li in lis:
    names.append( li[0] )
    na = li[0][6:-4]
    fe_path = cfg.fe_mel_eva_fd + '/' + na + '.f'
    X = cPickle.load( open( fe_path, 'rb' ) )
    X = mat_2d_to_3d( X, agg_num, hop )
    
    # predict
    p_y_preds = md.predict(X)        # probability, size: (n_block,label)
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    pred = int( b[0] )
    pred_lbs.append( cfg.id_to_lb[ pred ] )
    
# write out result
f = open('Results/task1_results.txt', 'w')
for i1 in xrange( len( names ) ):
    f.write( names[i1] + '\t' + pred_lbs[i1] + '\n' )
f.close()
print 'write out finished!'