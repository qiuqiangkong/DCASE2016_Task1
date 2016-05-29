'''
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Training time: 6 s/epoch. (Tesla M2090)
          test acc: 77% +- ?, test frame acc: 63% +- ?  after 50 epoches     
          Try adjusting hyper-params, optimizer, longer epoches to get better results. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.05.28
--------------------------------------
'''
import sys
sys.path.append('../Hat')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import SGD, Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
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
md.add( InputLayer( (n_time, n_freq), name='lay0' ) )
md.add( Flatten() )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act, name='lay1' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act, name='lay2' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_hid, act=act, name='lay3' ) )
md.add( Dropout( 0.1 ) )
md.add( Dense( n_out, act='softmax', name='lay4' ) )
md.summary()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
# optimizer = SGD( 0.01, 0.95 )
optimizer = Rmsprop(0.001)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=100, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 