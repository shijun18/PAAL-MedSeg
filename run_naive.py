import os
import argparse
import time
import random
import json

from trainer_naive import SemanticSeg
from config_naive import INIT_TRAINER, SETUP_TRAINER, CURRENT_FOLD, SPLIT_JSON, FOLD_NUM


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross'],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()
    random.seed(0)
    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
        
    split_json = SPLIT_JSON
    with open(split_json,'r') as fp:
        data_list = json.load(fp)
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path = data_list[f'fold{current_fold}']['train_path']
            val_path = data_list[f'fold{current_fold}']['val_path']
            SETUP_TRAINER['args'] = {**INIT_TRAINER, **SETUP_TRAINER} 
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        print("=== Training Fold ", CURRENT_FOLD, " ===")
        train_path = data_list[f'fold{CURRENT_FOLD}']['train_path']
        val_path = data_list[f'fold{CURRENT_FOLD}']['val_path']
        SETUP_TRAINER['args'] = {**INIT_TRAINER, **SETUP_TRAINER} 
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################