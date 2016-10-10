'''
SUMMARY:  calculate all features for evaluation data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
Modified: 2016.10.09 modify variable name
--------------------------------------
'''
import prepare_dev_data as pp_dev_data
import os
import config as cfg

def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)

if __name__ == "__main__":
    # create folders
    pp_dev_data.CreateFolder( cfg.eva_fe_fd )
    pp_dev_data.CreateFolder( cfg.eva_fe_mel_fd )
    
    # calculate mel feature
    pp_dev_data.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=0 )