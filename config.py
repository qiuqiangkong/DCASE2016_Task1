'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.06.24 Delete unnecessary
          2016.10.09 rename some variables
--------------------------------------
'''
# development config
dev_wav_fd = '/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/audio'

dev_csv_fd = '/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/evaluation_setup'
dev_tr_csv = [ dev_csv_fd+'/fold1_train.txt', dev_csv_fd+'/fold2_train.txt', 
               dev_csv_fd+'/fold3_train.txt', dev_csv_fd+'/fold4_train.txt' ]
dev_te_csv = [ dev_csv_fd+'/fold1_evaluate.txt', dev_csv_fd+'/fold2_evaluate.txt', 
               dev_csv_fd+'/fold3_evaluate.txt', dev_csv_fd+'/fold4_evaluate.txt' ]
dev_meta_csv = '/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-development/meta.txt'

# evaluation config
eva_wav_fd = '/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-evaluation/audio'

eva_txt_path = '/vol/vssp/datasets/audio/dcase2016/task1/TUT-acoustic-scenes-2016-evaluation/evaluation_setup/test.txt'

# your workspace
scrap_fd = "/vol/vssp/msos/qk/DCASE2016_task1_scrap"
dev_fe_fd = scrap_fd + '/Fe_dev'
dev_fe_mel_fd = scrap_fd + '/Fe_dev/Mel'
eva_fe_fd = scrap_fd + '/Fe_eva'
eva_fe_mel_fd = scrap_fd + '/Fe_eva/Mel'
dev_md = scrap_fd + '/Md_dev'
eva_md = scrap_fd + '/Md_eva'

# 1 of 15 acoustic label
labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
            
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }