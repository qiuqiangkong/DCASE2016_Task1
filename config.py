'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
wav_fd = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-development/audio'
fe_mel_fd = 'Fe/Mel'
fe_fft_fd = 'Fe/Fft'
fe_mel3d_fd = 'Fe/Mel3d'
fe_texture3d0_fd = 'Fe/Texture3d0'
fe_texture3d90_fd = 'Fe/Texture3d90'

csv_fd = '/homes/qkong/datasets/DCASE2016/Scene Classification/TUT-acoustic-scenes-2016-development/evaluation_setup'
tr_csv = [ csv_fd+'/fold1_train.txt', csv_fd+'/fold2_train.txt', csv_fd+'/fold3_train.txt', csv_fd+'/fold4_train.txt' ]
te_csv = [ csv_fd+'/fold1_evaluate.txt', csv_fd+'/fold2_evaluate.txt', csv_fd+'/fold3_evaluate.txt', csv_fd+'/fold4_evaluate.txt' ]

# 1 of 15 acoustic label
labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
            
# 1 of 3 scene label
labels2 = [ 'vehicle', 'indoor', 'outdoor' ]

# acoustic to scene
acoustic_to_scene = { 'bus':'vehicle', 'cafe/restaurant':'indoor', 'car':'vehicle', 'city_center':'outdoor', 
                      'forest_path':'outdoor', 'grocery_store':'indoor', 'home':'indoor', 'beach':'outdoor', 
                      'library':'indoor', 'metro_station':'indoor', 'office':'indoor',
                      'residential_area':'outdoor', 'train':'vehicle', 'tram':'vehicle', 'park':'outdoor'  }
            
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }
lb2_to_id = { lb2:id for id, lb2 in enumerate(labels2) }
id_to_lb2 = { id:lb2 for id, lb2 in enumerate(labels2) }