'''
SUMMARY:  Train DNN model on private dataset
          Training time: 9 s/epoch. (Tesla M2090)
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
Modified: 2016.10.09 Modify variable names
--------------------------------------
'''
import pickle
import os
import numpy as np
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.layers.cnn import Convolution2D
from hat.layers.rnn import LSTM, GRU
from hat.layers.pool import MaxPool2D, GlobalMeanTimePool
from hat.preprocessing import sparse_to_categorical, reshape_3d_to_4d
from hat.optimizers import SGD, Rmsprop, Adam
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data

# hyper-params
fe_fd = cfg.dev_fe_mel_fd   # use development data for training
agg_num = 11        # concatenate frames
hop = 5            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )

# prepare data
scaler = pp_dev_data.Scaler( fe_fd, cfg.dev_meta_csv )
tr_X, tr_y = pp_dev_data.GetAllData( fe_fd, cfg.dev_meta_csv, agg_num, hop, scaler )
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
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, call_freq=1, dump_path=None )

# save model
if not os.path.exists( cfg.eva_md ): os.makedirs( cfg.eva_md )
save_model = SaveModel( dump_fd=cfg.eva_md, call_freq=5 )

# callbacks
callbacks = [ validation, save_model ]

# optimizer
optimizer = Adam(1e-3)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epochs=100, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )