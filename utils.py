import os,glob
import pandas as pd 
import h5py
import SimpleITK as sitk
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance

def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.nanmean(dice_list),4)


def hd_2d(true,pred):
    hd_list = []
    for i in range(true.shape[0]):
        if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
            hd_list.append(hausdorff_distance(true[i],pred[i]))
    
    return np.mean(hd_list)

def multi_hd(y_true,y_pred,num_classes):
    hd_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        hd = hd_2d(true,pred)
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.nanmean(hd_list),4)


def get_path_with_column(input_path,path_col):
    final_list = pd.read_csv(input_path)[path_col].values.tolist()
    return final_list


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list

def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5,reversed_flag=False):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if reversed_flag:
        if int(len(without_list)/ratio) < len(with_list):
            random.shuffle(with_list)
            with_list = with_list[:int(len(without_list)/ratio)]
    else:
        if int(len(with_list)/ratio) < len(without_list):
            random.shuffle(without_list)
            without_list = without_list[:int(len(with_list)/ratio)]    
        
    return with_list + without_list


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image



def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print(sample_list)
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


def rename_weight_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        for pth in os.scandir(ckpt_path):
            if ':' in pth.name:
                new_pth = pth.path.replace(':','=')
                print(pth.name,' >>> ',os.path.basename(new_pth))
                os.rename(pth.path,new_pth)
            else:
                break


def dfs_rename_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_rename_weight(sub_path.path)
        else:
            rename_weight_path(ckpt_path)
            break  


def print_dict_items(dict_data):
    max_key_length = max(len(key) for key in dict_data)
    for key, value in dict_data.items():
        print(f'{key.ljust(max_key_length)}: {value}')

if __name__ == "__main__":
    # input_path = "/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/lung.csv"
    # input_path = "/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/segthor.csv"
    # path_list = get_path_with_annotation(input_path,'path','Lung-R')
    # path_list = os.listdir('/staff/shijun/dataset/Med_Seg/LITS/2d_data')
    # _,_ = get_cross_validation_by_sample(path_list,5,1)

    # ckpt_path = './ckpt/Lung/2d_clean/v1.3'
    # ckpt_path = './new_ckpt/Nasopharynx/2d/v4.1/All/fold4'
    # dfs_remove_weight(ckpt_path)
    # ckpt_path = './new_ckpt/Stomach'
    # dfs_rename_weight(ckpt_path)

    ckpt_path = './new_ckpt/Liver'
    dfs_rename_weight(ckpt_path)

