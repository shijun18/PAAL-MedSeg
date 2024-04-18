import os
import glob
import random
import json

json_path = {
    'Liver':'../converter/dcm_converter/static_files/Liver_Oar.json',
    'SegTHOR':'../converter/nii_converter/static_files/SegTHOR.json',
    'ACDC':'../converter/nii_converter/static_files/ACDC.json',
    'MSD01_BrainTumour':'../converter/nii_converter/static_files/MSD01_BrainTumour.json', # multi-modality
}


def get_cross_validation_by_sample_v2(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    print(train_id)
    print(validation_id)

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = []
    
    # customed for specific dataset
    for case in path_list:
        ID = os.path.basename(case).split('_')[:-1]
        sample_list.append('_'.join(ID))
   
    sample_list = list(set(sample_list))
    sample_list.sort()
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])
    
    print(train_id)
    print(validation_id)

    train_path = []
    validation_path = []
    for case in path_list:
        if '_'.join(os.path.basename(case).split('_')[:-1]) in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path


DATASET = 'ACDC'

random.seed(0)

with open(json_path[DATASET], 'r') as fp:
    info = json.load(fp)

PATH_LIST = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))

save_path = f'../dataset/{DATASET}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

split_dict = {}

for i in range(1,6):
    # train_path, val_path = get_cross_validation_by_sample(PATH_LIST,5,i)
    train_path, val_path = get_cross_validation_by_sample_v2(PATH_LIST,5,i)
    split_dict[f'fold{i}'] = {}
    split_dict[f'fold{i}']['val_path'] = val_path
    split_dict[f'fold{i}']['train_path'] = train_path

with open(os.path.join(save_path,'split.json'),'w') as fp:
    json.dump(split_dict,fp,indent=4)
