# weight object function, DNN
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential, Base
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.layers.cnn import Convolution2D
from Hat.layers.rnn import LSTM, GRU
from Hat.layers.pool import MaxPool2D, GlobalMeanTimePool
from Hat.preprocessing import sparse_to_categorical, reshape_3d_to_4d
from Hat.optimizers import SGD, Rmsprop
from Hat.supports import *
import Hat.backend as K
import config as cfg
import prepareData as ppData
import theano.tensor as T
import time

def my_loss( p_y_pred, y_gt ):
    _EPSILON = 1e-6
    shape = p_y_pred.shape
    p_y_pred = K.clip( p_y_pred, _EPSILON, 1. - _EPSILON )
    p_y_pred_flatten = p_y_pred.reshape( ( K.prod(shape[0:-1]), shape[-1] ) )
    y_gt_flatten = y_gt.reshape( ( K.prod(shape[0:-1]), shape[-1] ) )
    return T.nnet.categorical_crossentropy( p_y_pred_flatten, y_gt_flatten ).reshape( shape[0:-1] )

class Model2( Base ):
    def __init__( self, in_layers, out_layers, gate_layers, obj_weights=[1.] ):
        super( Model2, self ).__init__( in_layers )
        assert len(out_layers)==len(obj_weights), "num of out_layers must equal num of obj_weights!"
        
        # out layers
        out_layers = to_list( out_layers )
        self._out_layers = out_layers
        self._gate_layers = gate_layers
        self._obj_weights = obj_weights
        
        # out_nodes & create gt_nodes
        self._out_nodes = [ layer.output for layer in self._out_layers ]
        self._gt_nodes = [ K.placeholder( len(layer.out_shape) ) for layer in self._out_layers ]
        self._gate_nodes = [ layer.output for layer in self._gate_layers ]
        
        
        
    '''
    Fit model. x, y can be list of ndarrays. 
    '''
    def fit( self, x, y, batch_size=100, n_epoch=10, loss_type='categorical_crossentropy', optimizer=SGD( lr=0.01, rho=0.9 ), clip=None, 
             callbacks=[], verbose=1 ):
        x = to_list( x )
        y = to_list( y )
        
        # format
        x = [ K.format_data(e) for e in x ]
        y = [ K.format_data(e) for e in y ]
        
        # shuffle data
        x, y = shuffle( x, y )
        
        # check data
        self._check_data( y, loss_type )
        
        # memory usage
        mem_usage = memory_usage( x, y )
        print 'memory usage:', mem_usage / 8e6, 'Mb'
        
        # store data in shared memory (GPU)
        sh_x = [ K.sh_variable( value=e, name='tr_x' ) for e in x ]
        sh_y = [ K.sh_variable( value=e, name='tr_y' ) for e in y ]
        
        # loss
        pred_node = self._out_nodes[0]
        gt_node = self._gt_nodes[0]
        gate_node = self._gate_nodes[0]
        loss_node = K.mean( my_loss( pred_node, gt_node ) * gate_node )
        
        # gradient
        gparams = K.grad( loss_node + self._reg_value, self._params )
        
        # todo clip gradient
        if clip is not None:
            gparams = [ K.clip( gparam, -clip, clip ) for gparam in gparams ]
        
        # gradient based opt
        updates = optimizer.get_updates( self._params, gparams )
        
        # compile model
        input_nodes = self._in_nodes + self._gt_nodes
        output_nodes = [ loss_node ]
        given_nodes = sh_x + sh_y
        f = K.function_given( batch_size, input_nodes, self._tr_phase_node, output_nodes, given_nodes, updates )
        
        # debug
        # you can write debug function here
        #f_debug = K.function_no_given( self._in_nodes, self._layer_list[1].tmp )
        
        # compile for callback
        if callbacks is not None:
            callbacks = to_list( callbacks )
            for callback in callbacks:
                callback.compile( self ) 

        # train
        N = len( x[0] )
        batch_num = int( N / batch_size )
        while self._epoch < n_epoch:
            
            '''
            in_list = x+[0.]
            np.set_printoptions(threshold=np.nan, linewidth=1000, precision=10, suppress=True)
            print f_debug(*in_list)
            pause
            '''
            
            self.evaluate( x, y )
            # callback
            for callback in callbacks:
                if ( self._epoch % callback.call_freq == 0 ):
                    callback.call()

            print
            # train
            t1 = time.time()
            for i2 in xrange(batch_num):
                loss = f(i2, 1.)[0]                     # training phase          
                if verbose: self.print_progress( self._epoch, batch_num, i2 )
            t2 = time.time()
            self._tr_time += (t2 - t1)
            self._epoch += 1
            print
            #print '\n', t2-t1, 's'          # print an empty line
        
    def evaluate( self, x, y ):
        # format data
        x = to_list( x )
        x = [ K.format_data(e) for e in x ]
        y = to_list( y )
        y = [ K.format_data(e) for e in y ]
        
        # compile predict model
        if not hasattr( self, '_f_predict' ):
            if len( self._out_nodes )==1:   # if only 1 out_node, then return it directly instead of list
                print 'asdf'
                pred_node = self._out_nodes[0]
                gt_node = self._gt_nodes[0]
                gate_node = self._gate_nodes[0]
                loss_mat = pred_node * gate_node[:,None]
                self._f_predict = K.function_no_given( self._in_nodes, self._tr_phase_node, [loss_mat] )
                
            else:
                self._f_predict = K.function_no_given( self._in_nodes, self._tr_phase_node, self._out_nodes )
        
        # do predict
        in_list = x + [0.]
        y_out = self._f_predict( *in_list )[0]
        #y_out = np.sum( y_out, 1 )
        y_pred = np.argmax( y_out, axis=1 )
        gt = y[0]
        #gt = gt[:,0,:]
        gt = np.argmax( gt, axis=1 )
        print np.sum( np.not_equal(gt,y_pred) ) / float(len(x[0]))
        
        

# hyper-params
fe_fd = cfg.fe_mel3d_fd
agg_num = 10        # concatenate frames
hop = 10            # step_len
act = 'relu'
n_hid = 500
n_out = len( cfg.labels )

# prepare data
#tr_X, tr_y = ppData.Load4dData( fe_fd, cfg.tr_csv[0] )
#te_X, te_y = ppData.Load4dData( fe_fd, cfg.te_csv[0] )
tr_X, tr_y = ppData.LoadDataToMat( fe_fd, cfg.tr_csv[0] )
te_X, te_y = ppData.LoadDataToMat( fe_fd, cfg.te_csv[0] )
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

[_, n_time, n_freq] = tr_X.shape
print 'tr_X.shape:', tr_X.shape     # (batch_num, n_time, n_freq)
print 'tr_y.shape:', tr_y.shape     # (batch_num, n_labels )

# build model

x0 = InputLayer( (n_time, n_freq) )
x1 = Flatten( ndim=2 )( x0 )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
x2 = Dropout( 0.1 )( x1 )
x3 = Dense( 500, act=act )( x2 )
x4 = Dropout( 0.1 )( x3 )
x5 = Dense( 400, act=act )( x4 )
x6 = Dropout( 0.1 )( x5 )
x7 = Dense( 300, act=act )( x6 )
x8 = Dropout( 0.1 )( x7 )
x9 = Dense( n_out, act='softmax' )(x8)


g1 = Flatten( ndim=2 )( x0 )             # flatten to 2d: (n_time, n_freq) to 1d:(n_time*n_freq)
g2 = Dropout( 0.1 )( g1 )
g3 = Dense( 200, act=act )( g2 )
g4 = Dropout( 0.1 )( g3 )
g5 = Dense( 200, act=act )( g4 )
g6 = Dropout( 0.1 )( g5 )
g7 = Dense( 1, act='sigmoid' )( g6 )
g8 = Flatten(1)( g7 )


md = Model2( [x0], [x9], [g8] )
md.summary()
#pause

# callbacks
# tr_err, te_err are frame based. To get event based err, run recognize.py
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, dump_path='Results/validation.p' )
save_model = SaveModel( dump_fd='Md', call_freq=5 )
callbacks = []

# optimizer
# optimizer = SGD( 0.01, 0.95 )
optimizer = Rmsprop(1e-4)

# fit model
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epoch=1000, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

# run recognize.py to get results. 