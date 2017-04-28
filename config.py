"""
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.06.24 Delete unnecessary
          2016.10.09 rename some variables
          2017.04.27 Minor fix. 
--------------------------------------
"""
# hat deep learning toolbox
hat_root = ""   # you need to specify the source code path if hat is not 
                # installed on your computer. 

# development config
dev_wav_fd = "/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/audio"

dev_csv_fd = "/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/evaluation_setup"
dev_tr_csv = [dev_csv_fd+"/fold1_train.txt", dev_csv_fd+"/fold2_train.txt", 
               dev_csv_fd+"/fold3_train.txt", dev_csv_fd+"/fold4_train.txt"]
dev_te_csv = [dev_csv_fd+"/fold1_evaluate.txt", dev_csv_fd+"/fold2_evaluate.txt", 
               dev_csv_fd+"/fold3_evaluate.txt", dev_csv_fd+"/fold4_evaluate.txt"]
dev_meta_csv = "/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/meta.txt"

# evaluation config
eva_wav_fd = "/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-evaluation/audio"
eva_meta_csv = "/vol/vssp/AP_datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-evaluation/meta.txt"

# your workspace
scrap_fd = "/vol/vssp/msos/qk/DCASE2016_task1_scrap"
fe_fd = scrap_fd + "/features"
dev_fe_logmel_fd = fe_fd + "/dev/logmel"
eva_fe_logmel_fd = fe_fd + "/eva/logmel"
md_fd = scrap_fd + "/models"
dev_md_fd = scrap_fd + "/dev"
eva_md_fd = scrap_fd + "/eva"

# 1 of 15 acoustic label
labels = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 
           'grocery_store', 'home', 'beach', 'library', 'metro_station', 
           'office', 'residential_area', 'train', 'tram', 'park']
            
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}

fs = 44100.
n_fft = 1024.