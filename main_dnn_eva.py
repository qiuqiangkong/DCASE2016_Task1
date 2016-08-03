'''
SUMMARY:  Train DNN model on private dataset
          Training time: 9 s/epoch. (Tesla M2090)
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
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
from Hat.optimizers import SGD, Rmsprop, Adam
import Hat.backend as K
import config as cfg
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd   # use development data for training
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )

# prepare data
scaler = ppData.Scaler( fe_fd, cfg.meta_csv )
tr_X, tr_y = ppData.GetAllData( fe_fd, cfg.meta_csv, agg_num, hop, scaler )
tr_y = sparse_to_categorical( tr_y, n_out )

[batch_num, n_time, n_freq] = tr_X.shape
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )

# build model
seq = Sequential()
seq.add( InputLayer( (n_time, n_freq) ) )
seq.add( Flatten() )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( 500, act=act ) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( 500, act=act) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( 500, act=act) )
seq.add( Dropout( 0.1 ) )
seq.add( Dense( n_out, act='softmax' ) )
md = seq.combine()

md.summary()

# callbacks
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md_eva', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
optimizer = Adam(1e-4)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epoch=101, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )