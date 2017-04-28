"""
SUMMARY:  Dcase 2016 Task 1. Scene classification
          Train and test on whole dev dataset and evaluate on private dataset. 
          Training time: 20 s/epoch. (GTX TitanX)
          te_frame_based_acc: 81.0% +- ? , te_clip_based_acc: 69.2% +- ?
          after 10 epoches. 
AUTHOR:   Qiuqiang Kong
Created:  2017.04.28
Modified: - 
--------------------------------------
"""
import sys
import config as cfg
from main_dev_dnn import train, recognize

tr_fe_fd = cfg.dev_fe_logmel_fd
te_fe_fd = cfg.eva_fe_logmel_fd
n_concat = 11        # concatenate frames
hop = 5
n_out = len(cfg.labels)

if __name__ == '__main__':
    if sys.argv[1] == "--train": 
        train(tr_fe_fd=tr_fe_fd, 
              tr_csv_file=cfg.dev_meta_csv, 
              te_fe_fd=te_fe_fd, 
              te_csv_file=cfg.eva_meta_csv, 
              out_md_fd=cfg.eva_md_fd)
              
    elif sys.argv[1] == "--recognize":
        recognize(md_path=cfg.eva_md_fd+'/md10_epochs.p', 
                  tr_fe_fd=tr_fe_fd, 
                  tr_csv_ile=cfg.dev_meta_csv, 
                  te_fe_fd=te_fe_fd, 
                  te_csv_file=cfg.eva_meta_csv)
    else: 
        raise Exception("Incorrect argv!")