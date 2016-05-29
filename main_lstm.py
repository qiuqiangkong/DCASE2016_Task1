'''
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Training time: 12 s/epoch. (Tesla M2090)
          test acc: 70% +- ?, test frame acc: 56% +- ? after 10 epoches. 
          Try adjusting hyper-params, optimizer, longer epoches to get better results. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.05.29
--------------------------------------
'''
import sys
sys.path.append('../Hat')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.layers.rnn import SimpleRnn, LSTM, GRU
from Hat.layers.pool import GlobalMeanTimePool
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData


# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
n_hid = 500
n_out = len( cfg.labels )

# prepare data
trDict = ppData.GetDictData( fe_fd, cfg.tr_csv[0], agg_num, hop )
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[0], agg_num, hop )
tr_X, tr_y, _ = ppData.DictToMat( trDict )
te_X, te_y, _ = ppData.DictToMat( teDict )
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

[batch_num, n_time, n_freq] = tr_X.shape
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )

# build model
md = Sequential()
md.add( InputLayer( (n_time, n_freq) ) )
md.add( LSTM( n_out=100, act='tanh' ) )       # output size: (batch_num, n_time, n_freq). Try SimpleRnn, GRU instead. 
md.add( GlobalMeanTimePool( masking=None ) )  # mean along time axis, output shape: (batch_num, n_freq)
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_out, act='softmax' ) )
md.summary()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
optimizer = Rmsprop(0.001)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=100, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 