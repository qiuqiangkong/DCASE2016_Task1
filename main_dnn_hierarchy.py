'''
SUMMARY:  Dcase 2016 Task 1. Scene classification
          hierarchy classification. (acoustic label, scene label) Yong Xu's PPT
          Training time: 6 s/epoch. (Tesla M2090)
          Acoustic label acc: 78% +- ?, Scene label acc (1-of-3): 99%
          test frame acc: not evaluated. after 100 epoches     
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
from Hat.models import Sequential, Model
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
n_hid = 500
n_out1 = len( cfg.labels )
n_out2 = len( cfg.labels2 )

# prepare data
trDict = ppData.GetDictData( fe_fd, cfg.tr_csv[0], agg_num, hop )
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[0], agg_num, hop )
tr_X, tr_y, tr_y2 = ppData.DictToMat( trDict )      # tr_y is 1-of-15 acoustic label, tr_y2 is 1-of-3 scene label
te_X, te_y, te_y2 = ppData.DictToMat( teDict )
tr_y = sparse_to_categorical( tr_y, n_out1 )
te_y = sparse_to_categorical( te_y, n_out1 )
tr_y2 = sparse_to_categorical( tr_y2, n_out2 )
te_y2 = sparse_to_categorical( te_y2, n_out2 )

[batch_num, n_time, n_freq] = tr_X.shape
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )

# build model
md = Sequential()
a0 = InputLayer( (n_time, n_freq) )
a1 = Flatten()( a0 )
a2 = Dense( n_hid, act='relu' )( a1 )
a3 = Dense( n_hid, act='relu' )( a2 )
tar1 = Dense( n_out1, act='softmax' )( a3 )
tar2 = Dense( n_out2, act='softmax' )( a3 )
md = Model( in_layers=[a0], out_layers=[tar1, tar2], obj_weights=[0.8, 0.2] )
md.summary()
md.plot_connection()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=[tr_y, tr_y2], va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer
# optimizer = SGD( 0.01, 0.95 )
optimizer = Rmsprop(0.001)

# fit model
md.fit( x=tr_X, y=[tr_y, tr_y2], batch_size=500, n_epoch=100, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )
