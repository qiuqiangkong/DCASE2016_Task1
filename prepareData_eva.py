'''
SUMMARY:  calculate all features for evaluation data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.24
Modified: -
--------------------------------------
'''
import prepareData as ppData
import os
import config as cfg

def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)

if __name__ == "__main__":
    # create folders
    ppData.CreateFolder('Fe')
    CreateFolder('Results')
    CreateFolder('Md')
    ppData.CreateFolder('Fe_eva/Mel')
    ppData.CreateFolder('Md_eva')
    
    # calculate mel feature
    ppData.GetMel( cfg.wav_eva_fd, cfg.fe_mel_eva_fd, n_delete=0 )