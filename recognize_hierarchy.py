'''
SUMMARY:  For evaluating hierarchy classification. (acoustic label, scene label) Yong Xu's PPT
          Load trained model and Evaluate on test dataset
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: -
--------------------------------------
'''
import sys
sys.path.append('../Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
import Hat.backend as K
import config as cfg
import prepareData as ppData

# hyper-params
fe_fd = cfg.fe_mel_fd
agg_num = 10
hop = 10
n_labels = len( cfg.labels )

# load model
md = pickle.load( open( 'Md/md95.p', 'rb' ) )

# load test data
teDict = ppData.GetDictData( fe_fd, cfg.te_csv[0], agg_num, hop )

def GetAcousticId( p_y_preds, ke ):
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    acoustic_id = int( b[0] )
    return acoustic_id

def GetSceneId( p_y_preds, ke ):
    preds = np.argmax( p_y_preds, axis=-1 )     # size: (n_block)
    b = scipy.stats.mode(preds)
    scene_id = int( b[0] )
    return scene_id

# do recognize and evaluation
n_labels = len( cfg.labels )
n_labels2 = len( cfg.labels2 )
acoustic_confM = np.zeros( (n_labels, n_labels) )      # confusion matrix
scene_confM = np.zeros( (n_labels2, n_labels2) )
acc_frs = []
for ke in teDict.keys():
    for X in teDict[ke]:
        [p_y_preds1, p_y_preds2] = md.predict(X)        # p_y_preds size: (n_block, 15)
                                                       # p_y_preds2 size: (n_block, 3)
        gt_acoustic_id = cfg.lb_to_id[ke]
        gt_scene_id = cfg.lb2_to_id[ cfg.acoustic_to_scene[ke] ]
        pred_acoustic_id = GetAcousticId( p_y_preds1, ke )
        pred_scene_id = GetSceneId( p_y_preds2, ke )
                                                       
        acoustic_confM[ gt_acoustic_id, pred_acoustic_id ] += 1
        scene_confM[ gt_scene_id, pred_scene_id ] += 1
        
acoustic_acc = np.sum( np.diag( np.diag( acoustic_confM ) ) ) / np.sum( acoustic_confM )
scene_acc = np.sum( np.diag( np.diag( scene_confM ) ) ) / np.sum( scene_confM )

print acoustic_confM
print scene_confM
print 'acoustic_acc:', acoustic_acc
print 'scene_acc:', scene_acc