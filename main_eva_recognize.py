'''
SUMMARY:  Do classification on private dataset and write out result. 
          Training time: 25 s/epoch. (GTX TitanX)
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
Modified: 2016.10.09 Modify variable names
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
from hat import serializations
from hat.optimizers import Rmsprop
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data
import csv
import os
import cPickle

# hyper-params
fe_fd = cfg.eva_fe_mel_fd
agg_num = 11    # this should be same as training procedure
hop = 5
n_labels = len( cfg.labels )

# load model
md = serializations.load( cfg.eva_md+'/md10.p' )

# get scaler
scaler = pp_dev_data.Scaler( fe_fd, cfg.eva_txt_path )

# load name of wavs to be classified
with open( cfg.eva_txt_path, 'rb') as f:
    reader = csv.reader(f)
    lis = list(reader)

# do classification for each file
names = []
pred_lbs = []

for li in lis:
    names.append( li[0] )
    na = li[0][6:-4]
    fe_path = cfg.eva_fe_mel_fd + '/' + na + '.f'
    X = cPickle.load( open( fe_path, 'rb' ) )
    X = scaler.transform( X )
    X = mat_2d_to_3d( X, agg_num, hop )
    
    # predict
    p_y_preds = md.predict(X)        # probability, size: (n_block,label)
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    pred = int( b[0] )
    pred_lbs.append( cfg.id_to_lb[ pred ] )
    
# write out result
if not os.path.exists( cfg.scrap_fd+'/Results' ): os.makedirs( cfg.scrap_fd+'/Results' )
f = open(cfg.scrap_fd+'/Results/task1_results.txt', 'w')
for i1 in xrange( len( names ) ):
    f.write( names[i1] + '\t' + pred_lbs[i1] + '\n' )
f.close()
print 'write out finished!'