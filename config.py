
import os
import json
import glob

from utils import print_dict_items,get_weight_path

__Dataset__ = ['ACDC','SegTHOR','Liver','AMOS-CT','SegTHOR','MSD01_BrainTumour']
__Segnet__ = ['unet','unet++','FPN','deeplabv3+']
__Encoder_name__ = ['resnet18','resnet34','resnet50','resnet50_dropout','resnet50_naive']

__Predictor__ = ['ap18']
__Al_mode__ = ['ap','ap+wps'] # used: __Al_mode__ = ['single','km'] 

__Sample_mode__ = ['uniform','linear']
__Sample_strategy__ = ['norm','iq'] 

__mode__ = ['2d-AL']

json_path = {
    'Liver':'./dataset/Liver/Liver_Oar.json',
    'SegTHOR':'./dataset/SegTHOR/SegTHOR.json',
    'ACDC':'./dataset/ACDC/ACDC.json',
    'MSD01_BrainTumour':'./dataset/MSD01_BrainTumour/MSD01_BrainTumour.json',# multi-modality
}


SAMPLE_TIMES_MAP = {
    'ACDC':[[5,10,5],[5,20,15],[5,50,20]],
    'SegTHOR':[[5,10,5],[5,20,15],[5,50,20]],
    'Liver':[[5,10,5],[5,20,15],[5,50,20]],
    'MSD01_BrainTumour':[[0.5,5,10],[0.5,10,15],[0.5,20,15]],
}


MODE = '2d-AL'
DATASET = 'Liver' 

NET_NAME = 'unet'
ENCODER_NAME = 'resnet50'
PREDICTOR_NAME = 'ap18'

AL_MODE = 'ap+wps'
INIT_PERCENT = 5
MAX_PERCENT = 50
SAMPLE_MODE = 'uniform'
SAMPLE_STRATEGY = 'norm'
SAMPLE_TIMES = 20

VERSION = f'{NET_NAME}-{PREDICTOR_NAME}-{AL_MODE}-i{INIT_PERCENT}m{MAX_PERCENT}-{SAMPLE_MODE}-{SAMPLE_STRATEGY}'

DEVICE = '1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = True if 'pretrain' in VERSION else False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


with open(json_path[DATASET], 'r') as fp:
    info = json.load(fp)

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None# or 1,2,...
NUM_CLASSES = info['annotation_num'] + 1 # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    if isinstance(ROI_NUMBER,list):
        NUM_CLASSES = len(ROI_NUMBER) + 1
        ROI_NAME = 'Part_{}'.format(str(len(ROI_NUMBER)))
        TARGET_NAMES = [info['annotation_list'][i - 1] for i in ROI_NUMBER]
    else:
        NUM_CLASSES = 2
        ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
        TARGET_NAMES = [ROI_NAME]
else:
    ROI_NAME = 'All'
    TARGET_NAMES = info['annotation_list']


try:
    SCALE = info['scale'][ROI_NAME]
except:
    SCALE = None

try:
    MEAN = info['mean_std']['mean']
    STD = info['mean_std']['std']
except:
    MEAN = STD = None
#--------------------------------- mode and data path setting
SPLIT_JSON = f'./dataset/{DATASET}/split.json'
#---------------------------------


#--------------------------------- others
INPUT_SHAPE_DICT = {
    'ACDC':(512,512),
    'SegTHOR':(512,512),
    'Liver':(512,512),
    'MSD01_BrainTumour':(256,256),
}

INPUT_SHAPE = INPUT_SHAPE_DICT[DATASET]
BATCH_SIZE = 64


CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DATASET,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))
SEG_WEIGHT_PATH = get_weight_path(os.path.join(CKPT_PATH, 'segnet'))
PREDICTOR_WEIGHT_PATH = get_weight_path(os.path.join(CKPT_PATH, 'predictor'))
print(SEG_WEIGHT_PATH)
print(PREDICTOR_WEIGHT_PATH)


CHANNEL_DICT = {
    'ACDC':1,
    'SegTHOR':1,
    'Liver':1,
    'MSD01_BrainTumour':4,
}

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'predictor_name':PREDICTOR_NAME,
  'lr':1e-3, 
  'n_epoch': 400,
  'warmup_epoch':10,
  'sample_inteval':5,
  'channels':CHANNEL_DICT[DATASET],
  'num_classes':NUM_CLASSES,
  'target_names':TARGET_NAMES,
  'max_percent':MAX_PERCENT/100.0,
  'init_percent':INIT_PERCENT/100.0, 
  'roi_number':ROI_NUMBER, 
  'scale':SCALE,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':4,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'seg_weight_path':SEG_WEIGHT_PATH,
  'predictor_weight_path':PREDICTOR_WEIGHT_PATH,
  'weight_decay': 1e-4,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [30,60,90],
  'mean':MEAN,
  'std':STD,
  'topk':20,
  'use_fp16':True, #False if the machine you used without tensor core
 }
#---------------------------------

__seg_loss__ = ['TopKLoss','DiceLoss','CEPlusDice','CELabelSmoothingPlusDice','OHEM','Cross_Entropy']
__lr_scheduler__= ['CosineAnnealingLR','CosineAnnealingWarmRestarts','MultiStepLR','CustomScheduler']
# Arguments when perform the trainer 


SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(DATASET,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(DATASET,MODE,VERSION,ROI_NAME),
  'optimizer':'AdamW',
  'seg_loss_fun':'CELabelSmoothingPlusDice',
  'predictor_loss_fun':'MSE',
  'sample_mode':SAMPLE_MODE,
  'al_mode':AL_MODE,
  'score_type':'log_mean', # 'mean','log_mean'
  'sample_from_all_data':True,
  'sample_weight':None,
  'class_weight':None,
  'lr_scheduler':'CustomScheduler', 
  'freeze_encoder':False,
  'get_roi': False if 'roi' not in VERSION else True,
  'repeat_factor':1.0,
  'sample_strategy':SAMPLE_STRATEGY,
  'sample_patience':10,
  'sample_times':SAMPLE_TIMES
}

print_dict_items({**INIT_TRAINER, **SETUP_TRAINER})
#---------------------------------

TEST_PATH = None


        