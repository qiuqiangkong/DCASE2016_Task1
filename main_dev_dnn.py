'''
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Training time: 17 s/epoch. (GTX TitanX)
          train frame acc: 90.0% +- ?, test frame acc: 61.4% +- ? , test event acc: 74.8% after 10 epoches on fold 0
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.05.28
          2016.08.01 Add normalization of data
          2016.10.09 modify variable names
--------------------------------------
'''
import pickle
import numpy as np
import os
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, reshape_3d_to_4d
from hat.optimizers import SGD, Rmsprop, Adam
import hat.backend as K
import config as cfg
import prepare_dev_data as pp_dev_data

# hyper-params
fe_fd = cfg.dev_fe_mel_fd
agg_num = 11        # concatenate frames
hop = 5            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )
fold = 0            # can be 0, 1, 2, 3

# prepare data
scaler = pp_dev_data.Scaler( fe_fd, cfg.dev_tr_csv[fold] )
tr_X, tr_y = pp_dev_data.GetAllData( fe_fd, cfg.dev_tr_csv[fold], agg_num, hop, scaler )
te_X, te_y = pp_dev_data.GetAllData( fe_fd, cfg.dev_te_csv[fold], agg_num, hop, scaler )
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

[batch_num, n_time, n_freq] = tr_X.shape
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )


# build model
seq = Sequential()
seq.add( InputLayer( (n_time, n_freq) ) )
seq.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
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
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, call_freq=1, dump_path=None )

# save model
if not os.path.exists( cfg.dev_md ): os.makedirs( cfg.dev_md )
save_model = SaveModel( dump_fd=cfg.dev_md, call_freq=5 )

# callbacks
callbacks = [ validation, save_model ]

# optimizer
optimizer = Adam(1e-3)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epochs=1001, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 