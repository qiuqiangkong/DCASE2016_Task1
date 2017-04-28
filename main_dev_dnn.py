"""
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Train and test on dev dataset (split to 4 cv-folders)
          Training time: 17 s/epoch. (GTX TitanX)
          te_frame_based_acc: 61.4% +- ? , te_clip_based_acc: 80.7% +- ?
          after 10 epoches on fold 0. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.05.28
          2016.08.01 Add normalization of data. 
          2016.10.09 Modify variable names. 
          2017.04.27 Rewrite. 
--------------------------------------
"""
import cPickle
import numpy as np
import os
import sys
import csv
import config as cfg
import prepare_data as pp_data
sys.path.append(cfg.hat_root)
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from hat.optimizers import SGD, Adam
import hat.backend as K
from hat import serializations

# hyper-params
fe_fd = cfg.dev_fe_logmel_fd
n_concat = 11        # concatenate frames
hop = 5            # step_len
n_out = len(cfg.labels)
fold = 0            # can be 0, 1, 2, 3

def train(tr_fe_fd, tr_csv_file, te_fe_fd, te_csv_file, out_md_fd):
    # Prepare data
    scaler = pp_data.get_scaler(fe_fd=tr_fe_fd, 
                               csv_file=tr_csv_file, 
                               with_mean=True, 
                               with_std=True)
                               
    tr_x, tr_y = pp_data.get_matrix_format_data(
                     fe_fd=tr_fe_fd, 
                     csv_file=tr_csv_file, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
                     
    te_x, te_y = pp_data.get_matrix_format_data(
                     fe_fd=te_fe_fd, 
                     csv_file=te_csv_file, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
    
    n_freq = tr_x.shape[2]
    print 'tr_x.shape:', tr_x.shape     # (n_samples, n_concat, n_freq)
    print 'tr_y.shape:', tr_y.shape     # (n_samples, n_labels)
    
    
    # Build model
    seq = Sequential()
    seq.add(InputLayer((n_concat, n_freq)))
    seq.add(Flatten())
    seq.add(Dropout(0.2))
    seq.add(Dense(200, act='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(200, act='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(n_out, act='softmax'))
    md = seq.compile()
    md.summary()
    
    # Validation. 
    # tr_err, te_err are frame based. To get event based err, run recognize.py
    validation = Validation(tr_x=tr_x, tr_y=tr_y, 
                            va_x=None, va_y=None, 
                            te_x=te_x, te_y=te_y, 
                            batch_size=500, call_freq=1, dump_path=None)
    
    # Save model
    pp_data.create_folder(out_md_fd)
    save_model = SaveModel(out_md_fd, call_freq=2)
    
    # Callbacks
    callbacks = [validation, save_model]
    
    # Optimizer
    optimizer = Adam(1e-3)
    
    # fit model
    md.fit(x=tr_x, y=tr_y, 
           batch_size=100, 
           n_epochs=101, 
           loss_func='categorical_crossentropy', 
           optimizer=optimizer, 
           callbacks=callbacks)
           
def recognize(md_path, tr_fe_fd, tr_csv_ile, te_fe_fd, te_csv_file):
    # Load model
    md = serializations.load(md_path)
    
    # Get scaler
    scaler = pp_data.get_scaler(fe_fd=tr_fe_fd, 
                                csv_file=tr_csv_ile, 
                                with_mean=True, with_std=True)
    
    # Recognize and get statistics
    n_labels = len(cfg.labels)
    confuse_mat = np.zeros((n_labels, n_labels))      # confusion matrix
    frame_based_accs = []
    
    # Get test file names
    with open(te_csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    # Predict for each scene
    for li in lis:
        # Load data
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = te_fe_fd + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        x = scaler.transform(x)
        x = mat_2d_to_3d(x, n_concat, hop)
    
        # Predict
        p_y_preds = md.predict(x)[0]        # (n_block,label)
        pred_ids = np.argmax(p_y_preds, axis=-1)     # (n_block,)
        pred_id = int(pp_data.get_mode_value(pred_ids))
        gt_id = cfg.lb_to_id[lb]
        
        # Statistics
        confuse_mat[gt_id, pred_id] += 1            
        n_correct_frames = list(pred_ids).count(gt_id)
        frame_based_accs += [float(n_correct_frames) / len(pred_ids)]
            
    clip_based_acc = np.sum(np.diag(np.diag(confuse_mat))) / np.sum(confuse_mat)
    frame_based_acc = np.mean(frame_based_accs)
    
    print 'event_acc:', clip_based_acc
    print 'frame_acc:', frame_based_acc
    print confuse_mat

if __name__ == '__main__':
    if sys.argv[1] == "--train": 
        train(tr_fe_fd=fe_fd, 
              tr_csv_file=cfg.dev_tr_csv[fold], 
              te_fe_fd=fe_fd, 
              te_csv_file=cfg.dev_te_csv[fold], 
              out_md_fd=cfg.dev_md_fd)
              
    elif sys.argv[1] == "--recognize":
        recognize(md_path=cfg.dev_md_fd+'/md10_epochs.p', 
                  tr_fe_fd=fe_fd, 
                  tr_csv_ile=cfg.dev_tr_csv[fold], 
                  te_fe_fd=fe_fd, 
                  te_csv_file=cfg.dev_te_csv[fold])
    else: 
        raise Exception("Incorrect argv!")