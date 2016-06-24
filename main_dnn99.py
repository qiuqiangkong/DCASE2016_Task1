'''
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Training time: 3 s/epoch. (Tesla M2090)
          test acc: 78% +- ?, test frame acc: 64% +- ?  after 100 epoches     
          fold, agg_num, hop, n_hid, act, optimizer can be tuned. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.05.28
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.layers.cnn import Convolution2D
from Hat.layers.rnn import LSTM, GRU
from Hat.layers.pool import MaxPool2D, GlobalMeanTimePool
from Hat.preprocessing import sparse_to_categorical, reshape_3d_to_4d
from Hat.optimizers import SGD, Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData
from sklearn import preprocessing

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )
fold = 0            # can be 0, 1, 2, 3
is_load_md = True

# prepare data
trDict = ppData.GetDictData( fe_fd, cfg.tr_csv[fold], agg_num, hop )
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[fold], agg_num, hop )
#trDict = ppData.GetDictData0( fe_fd, cfg.tr_csv[fold] )
#teDict = ppData.GetDictData0( fe_fd, cfg.te_csv[fold] )
tr_X, tr_y, _ = ppData.DictToMat( trDict )
te_X, te_y, _ = ppData.DictToMat( teDict )
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1]*tr_X.shape[2])
te_X = te_X.reshape(te_X.shape[0], te_X.shape[1]*te_X.shape[2])
print tr_X.shape

n_in = tr_X.shape[1]
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )
scalar = preprocessing.StandardScaler().fit(tr_X)
tr_X = scalar.transform( tr_X )
te_X = scalar.transform( te_X )
pickle.dump( scalar, open( 'Results/scalar.p', 'wb' ) )


if is_load_md:
    md = pickle.load( open( 'Md/md110.p', 'rb' ) )
else:
    # build model
    md = Sequential()
    md.add( InputLayer( (n_in) ) )
    md.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
    md.add( Dropout( 0.1 ) )
    md.add( Dense( 500, act=act ) )
    md.add( Dropout( 0.1 ) )
    md.add( Dense( 500, act=act) )
    md.add( Dropout( 0.1 ) )
    md.add( Dense( 500, act=act) )
    md.add( Dropout( 0.1 ) )
    md.add( Dense( n_out, act='sigmoid' ) )
    md.summary()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
# optimizer = SGD( 1e-5, 0.95 )
optimizer = Rmsprop(1e-4)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=1000, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 