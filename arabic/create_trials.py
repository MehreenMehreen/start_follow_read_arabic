#!/usr/bin/env python
# coding: utf-8

# # Create different splits of train, valid, test sets
# # Is hard coded to split the public dataset with 1100 training images


import sys
import json
import os
import numpy as np
import yaml
import shutil
import create_set_routines as create



def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return config       


def write_json_files(directory, json_sets, total_train_valid):
    setnames = ['train_{}.json'.format(total_train_valid), 
                'valid_{}.json'.format(total_train_valid),
                'test_{}.json'.format(total_train_valid)]
    
    
    for s, json_obj in zip(setnames, json_sets):
        json_filename = os.path.join(directory, s)
        with open(json_filename, 'w') as fout:
            json_str = json.dumps(json_obj)
            fout.write(json_str)
            
def get_char_set_length(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)
    return len(char_set['idx_to_char'])

# Pretraining to start from init folder. Fine tuned network to go into pretrain folder    
def edit_pretrain_entries(config, set_dir, stop_after_no_improvement=15, images_per_epoch=100, PRETRAIN_SUFFIX="", 
                          lf_images_per_epoch=0):
    if lf_images_per_epoch == 0:
        lf_images_per_epoch = images_per_epoch

    # HW specific
    config['network']['hw']['char_set_path'] = 'trials/charset.json'
    config['network']['hw']['use_instance_norm'] = True
    config['network']['hw']['transfer_path'] = 'trials/pretrained/hw.pt'
    config['network']['hw']['num_of_outputs'] = get_char_set_length(config['network']['hw']['char_set_path']) + 1
    config['network']['hw']['input_height'] = 60

    # GEneral
    config['pretraining']['hw']['stop_after_no_improvement'] = stop_after_no_improvement
    config['pretraining']['sol']['stop_after_no_improvement'] = stop_after_no_improvement
    config['pretraining']['lf']['stop_after_no_improvement'] = stop_after_no_improvement
    
    config['pretraining']['lf']['images_per_epoch'] = lf_images_per_epoch
    config['pretraining']['sol']['images_per_epoch'] = images_per_epoch
    config['pretraining']['hw']['images_per_epoch'] = images_per_epoch
    
    config['pretraining']['lf']['log_file'] = os.path.join(set_dir, 'pretrain_lf_log.csv')
    config['pretraining']['sol']['log_file'] = os.path.join(set_dir, 'pretrain_sol_log.csv')
    config['pretraining']['hw']['log_file'] = os.path.join(set_dir, 'pretrain_hw_log.csv')
    
    config['pretraining']['snapshot_path'] = config['training']['snapshot']['pretrain']
    config['pretraining']['pretrained_path'] = config['training']['snapshot']['init']
    
    config['pretraining']['training_set']['file_list'] = os.path.join(set_dir, 'pretrain_train_{}.json'.format(PRETRAIN_SUFFIX))
    config['pretraining']['validation_set']['file_list'] = os.path.join(set_dir, 'pretrain_valid_{}.json'.format(PRETRAIN_SUFFIX))
    
    return config
    
  


# Split the entire json randomly (seed as input param) to train, valid, test sets
# Total train valid is array with [total_train_examples total_valid_examples]
# Also writes division file
def split_pretrain_json_and_write(all_json_files, train_valid, seed, set_dir):
    
    np.random.seed(seed)
    all_json = all_json_files.copy()
    shuffled_ind = np.random.permutation(len(all_json))
    all_json = [all_json_files[i] for i in shuffled_ind]
    all_json_count = len(all_json)
    
    total_train = train_valid[0]

    total_valid = train_valid[1]
    total_test = all_json_count - total_train - total_valid

    train_json = all_json[:total_train]
    valid_json = all_json[total_train:total_valid+total_train]
    test_json = all_json[total_valid+total_train:]
    total_train_valid = total_train + total_valid
    
        # Write files
    train_file = os.path.join(set_dir, 'pretrain_train_{}.json'.format(total_train_valid))
    valid_file = os.path.join(set_dir, 'pretrain_valid_{}.json'.format(total_train_valid))
    test_file = os.path.join(set_dir, 'pretrain_test_{}.json'.format(total_train_valid))
    
    for json_obj, file in zip([train_json, valid_json, test_json],
                              [train_file, valid_file, test_file]):
        with open(file, 'w') as fout:
            json.dump(json_obj, fout)
    
    division_file = set_dir + 'division_{}.txt'.format(total_train_valid)
    
    with open(division_file, 'w') as f:
        f.write(str(list(shuffled_ind)))
        f.write('\nSEED is {}\n'.format(seed))
        img_files = [(ind, f[1]) for ind, f in enumerate(all_json)]
        f.write(str(img_files))
    

def write_config_for_pretrain_only(config, set_dir, total_train_valid, copy_pretrained=False):
    config_filename = 'config'
    # Test file    
    config['testing'] = dict()
    config['testing']['test_file'] = os.path.join(set_dir, 'pretrain_test_{}.json'.format(total_train_valid))
    
        
    # Reset interval
    config['training']['lf']['reset_interval'] = 60*60*120  # 5 days
    config['training']['hw']['reset_interval'] = 60*60*120
    config['training']['sol']['reset_interval'] = 60*60*120
    
    # Stage 2 specific
    config['training']['alignment']['train_log_file'] = set_dir + f'log_align_train_{total_train_valid}.csv'
    config['training']['alignment']['validate_log_file'] = set_dir + f'log_align_validate_{total_train_valid}.csv'

    config['training']['hw']['log_file'] = set_dir + f'log_hw_{total_train_valid}.csv'
    config['training']['lf']['log_file'] = set_dir + f'log_lf_{total_train_valid}.csv'
    config['training']['sol']['log_file'] = set_dir + f'log_sol_{total_train_valid}.csv'

    config['training']['snapshot']['best_overall'] = set_dir + f'snapshot_{total_train_valid}/best_overall/'
    config['training']['snapshot']['best_validation'] = set_dir + f'snapshot_{total_train_valid}/best_validation/'
    config['training']['snapshot']['current'] = set_dir + f'snapshot_{total_train_valid}/current/'

    config['training']['training_set']['file_list'] = set_dir + f'train_{total_train_valid}'
    config['training']['validation_set']['file_list'] = set_dir + f'valid_{total_train_valid}'
    snapshot_folder = set_dir+'snapshot_{}/'.format(total_train_valid)
    if not os.path.exists(snapshot_folder):
        os.mkdir(snapshot_folder)
    # Individual networks to start training from 
    # Copy the networks in the respective current folder
    
    for folder in ['pretrain/', 'init/']:
        dst_folder = set_dir+'snapshot_{}/{}/'.format(total_train_valid, folder)
        config['training']['snapshot'][folder[:-1]] = os.path.join(set_dir, 
                                                                  'snapshot_{}/'.format(total_train_valid),
                                                                        folder)
    config = edit_pretrain_entries(config, set_dir, 
                                   stop_after_no_improvement=40, images_per_epoch=2000,
                                   PRETRAIN_SUFFIX=total_train_valid, lf_images_per_epoch=400)
    config_filename = os.path.join(set_dir, 
                                   config_filename + '_{}.yaml'.format(total_train_valid))
    with open(config_filename, 'w') as fout:
        yaml.dump(config, fout) 


if __name__ == "__main__":

    # This will create a reproducible split of train, valid, test sets            
    SEED = 37
    main_data_sfr_dir = 'data_files/sfr_arabic'
    if len(sys.argv) > 1:
        main_data_sfr_dir = sys.argv[1]
            
    input_pretrain_batch = ['public_sfr']
    # Will run on public dataset with 1100 training images
    output_dir = 'trials/public_1100/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    sets_todo = ['set0/', 'set1/', 'set2/']

    train_valid = [1100, 50]
    total_train_valid = train_valid[0] + train_valid[1]
    PRETRAIN_SUFFIX = total_train_valid
    sample_config = get_config('trials/sample_config.yaml')  

    for index, _ in enumerate(sets_todo):
        SEED = SEED+index*10
        set_dir = output_dir + sets_todo[index]
        # Create set dir
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)   
        # Get json for all files
        all_pretrain_json = create.get_all_data(main_data_sfr_dir, input_pretrain_batch)
        # Split json to train, test, valid and write them all. Also writes division file
        split_pretrain_json_and_write(all_pretrain_json, train_valid, SEED, set_dir)
        write_config_for_pretrain_only(sample_config, set_dir, total_train_valid, copy_pretrained=False)
        
        
    print('done') 
            
    













