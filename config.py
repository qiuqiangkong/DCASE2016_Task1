'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: 2016.06.24 Delete unnecessary
--------------------------------------
'''
# development config
wav_fd = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-development/audio'
fe_mel_fd = 'Fe/Mel'
csv_fd = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-development/evaluation_setup'
tr_csv = [ csv_fd+'/fold1_train.txt', csv_fd+'/fold2_train.txt', csv_fd+'/fold3_train.txt', csv_fd+'/fold4_train.txt' ]
te_csv = [ csv_fd+'/fold1_evaluate.txt', csv_fd+'/fold2_evaluate.txt', csv_fd+'/fold3_evaluate.txt', csv_fd+'/fold4_evaluate.txt' ]
meta_csv = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-development/meta.txt'

# evaluation config
wav_eva_fd = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-evaluation/audio'
fe_mel_eva_fd = 'Fe_eva/Mel'
txt_eva_path = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-evaluation/evaluation_setup/test.txt'

# 1 of 15 acoustic label
labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
            
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }