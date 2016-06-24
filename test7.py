# combining mel, texture feature train dnn
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential, Model
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout, Merge
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
fe_fd = cfg.fe_texture3d90_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )

# prepare data
tr_X1, tr_y = ppData.LoadDataToMat( cfg.fe_mel3d_fd, cfg.tr_csv[0] )
te_X1, te_y = ppData.LoadDataToMat( cfg.fe_mel3d_fd, cfg.te_csv[0] )
tr_X2, _ = ppData.LoadDataToMat( cfg.fe_texture3d0_fd, cfg.tr_csv[0] )
te_X2, _ = ppData.LoadDataToMat( cfg.fe_texture3d0_fd, cfg.te_csv[0] )
tr_X3, _ = ppData.LoadDataToMat( cfg.fe_texture3d90_fd, cfg.tr_csv[0] )
te_X3, _ = ppData.LoadDataToMat( cfg.fe_texture3d90_fd, cfg.te_csv[0] )

tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

isLoadMd = True
if isLoadMd is True:
    md = pickle.load( open( 'Md/md150.p', 'rb' ) )
    optimizer = SGD( 1e-5, 0.95 )
else:
    # build model
    a0 = InputLayer( tr_X1.shape[1:] )
    a1 = Flatten()( a0 )
    a2 = Dense( 200, act='relu' )( a1 )
    a3 = Dropout( 0.1 )( a2 )
    a4 = Dense( 200, act='relu' )( a3 )
    a5 = Dropout( 0.1 )(a4)
    
    b0 = InputLayer( tr_X2.shape[1:] )
    b1 = Flatten()( b0 )
    b2 = Dense( 200, act='relu' )( b1 )
    b3 = Dropout( 0.1 )( b2 )
    b4 = Dense( 200, act='relu' )( b3 )
    b5 = Dropout( 0.1 )(b4)
    
    c0 = InputLayer( tr_X3.shape[1:] )
    c1 = Flatten()( c0 )
    c2 = Dense( 200, act='relu' )( c1 )
    c3 = Dropout( 0.1 )( c2 )
    c4 = Dense( 200, act='relu' )( c3 )
    c5 = Dropout( 0.1 )(c4)
    
    x0 = Merge()( [a5, b5, c5] )
    x1 = Dense( 500, act='relu' )( x0 )
    x2 = Dropout( 0.1 )( x1 )
    x3 = Dense( n_out, act='softmax' )( x2 )
    md = Model( [a0, b0, c0], [x3] )
    
    optimizer = Rmsprop(1e-4)

md.summary()
md.plot_connection()

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=[tr_X1, tr_X2, tr_X3], tr_y=tr_y, va_x=None, va_y=None, te_x=[te_X1, te_X2, te_X3], te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = [ validation, save_model ]

# optimizer



# fit model
md.fit( x=[tr_X1, tr_X2, tr_X3], y=tr_y, batch_size=100, n_epoch=1000, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 