
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

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 100        # concatenate frames
hop = 100            # step_len
act = 'relu'
n_hid = 1000
n_out = len( cfg.labels )
fold = 0            # can be 0, 1, 2, 3

# prepare data
trDict = ppData.GetDictData( fe_fd, cfg.tr_csv[fold], agg_num, hop )
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[fold], agg_num, hop )
tr_X, tr_y, _ = ppData.DictToMat( trDict )
te_X, te_y, _ = ppData.DictToMat( teDict )
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

[batch_num, n_time, n_freq] = tr_X.shape
tr_X = reshape_3d_to_4d( tr_X )
te_X = reshape_3d_to_4d( te_X )
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )



md = Sequential()
md.add( InputLayer( (1, n_time, n_freq) ) )
md.add( Convolution2D( n_outfmaps=8, n_row=5, n_col=5, act='relu') )
md.add( MaxPool2D( pool_size=(3,3) ) )
md.add( Convolution2D( n_outfmaps=32, n_row=5, n_col=5, act='relu') )
md.add( MaxPool2D( pool_size=(3,3) ) )
md.add( Flatten() )
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
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epoch=1000, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 