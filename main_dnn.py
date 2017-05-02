"""
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Train and test on dev dataset (split to 4 cv-folders)
          Training time: 17 s/epoch. (GTX TitanX)
          Dev: te_frame_based_acc: 61.4% +- ? , te_clip_based_acc: 80.7% +- ? 
          (Train on fold 1,2,3 and validate on fold 0)
          Eva: te_frame_based_acc: 81.0% +- ? , te_clip_based_acc: 69.2% +- ?
          after 10 epoches. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2017.05.02 
--------------------------------------
"""
import cPickle
import numpy as np
np.random.seed(1515)
import os
import sys
import csv

import config as cfg
import prepare_data as pp_data

sys.path.append(cfg.hat_root)
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from hat.optimizers import SGD, Adam
import hat.backend as K


def train(tr_fe_fd, tr_csv_file, te_fe_fd, te_csv_file, 
          n_concat, hop, scaler, out_md_fd):
    # Prepare data 
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
    n_out = len(cfg.labels)
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
           


if __name__ == '__main__':
    # hyper-params
    n_concat = 11        # concatenate frames
    hop = 5            # step_len
    fold = 0            # can be 0, 1, 2, 3
    
    dev_fe_fd = cfg.dev_fe_logmel_fd
    eva_fd_fd = cfg.eva_fe_logmel_fd
    
    if sys.argv[1] == "--dev_train": 
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        train(tr_fe_fd=dev_fe_fd, 
              tr_csv_file=cfg.dev_tr_csv[fold], 
              te_fe_fd=dev_fe_fd, 
              te_csv_file=cfg.dev_te_csv[fold], 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md_fd=cfg.dev_md_fd)
              
    elif sys.argv[1] == "--dev_recognize":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        pp_data.recognize(md_path=cfg.dev_md_fd+'/md10_epochs.p', 
                  te_fe_fd=dev_fe_fd, 
                  te_csv_file=cfg.dev_te_csv[fold], 
                  n_concat=n_concat, 
                  hop=hop, 
                  scaler=scaler)
                  
    elif sys.argv[1] == "--eva_train": 
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
        train(tr_fe_fd=dev_fe_fd, 
              tr_csv_file=cfg.dev_meta_csv, 
              te_fe_fd=eva_fd_fd, 
              te_csv_file=cfg.eva_meta_csv, 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md_fd=cfg.eva_md_fd)
              
    elif sys.argv[1] == "--eva_recognize":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
        pp_data.recognize(md_path=cfg.eva_md_fd+'/md10_epochs.p', 
                  te_fe_fd=eva_fd_fd, 
                  te_csv_file=cfg.eva_meta_csv, 
                  n_concat=n_concat, 
                  scaler=scaler, 
                  hop=hop)
    else: 
        raise Exception("Incorrect argv!")