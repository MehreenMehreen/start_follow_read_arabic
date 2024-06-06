# This script will write the results csv files in all folders
import sys
sys.path.append('py3/')
sys.path.append('arabic/')

import json
import pandas as pd
import yaml
import decode_one_image as decode
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt


def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return config     

def split_and_get_predictions(config, img_file, result_df, region_type, gt, model_mode):
    print("splitting img", img_file)
    try:
        img1, img2, ht = decode.split_image_horizontal(img_file)
        out1 = decode.network_output(config, img1, model_mode)  
        out1, predicted_text1 = decode.decode_one_img_with_info(config, out1) 
        out2 = decode.network_output(config, img2, model_mode)  
        out2, predicted_text2 = decode.decode_one_img_with_info(config, out2) 

        out = decode.merge_out(out1, out2, (0, ht))
        # Put together all gt for lines
        predicted_text1.extend(predicted_text2)
        
    except RuntimeError as e:
        
        print('*****FINAL ROUND: Could not process', img_file)
        predicted_text1 = '-1'
        out = {'sol':np.array([-1]), 'beginning':np.array([-1]), 'ending':np.array([-1]), 
               'lf':''}
        
    lf_points = str([l.tolist() for l in out['lf']])
    result_df.loc[len(result_df.index)] = [img_file, gt, predicted_text1, region_type,
                                          lf_points,
                                          str(out['beginning'].tolist()), str(out['ending'].tolist()),
                                          str(out['sol'].tolist())]

        
    return result_df
    
def get_predictions(config, image, gt, region_type, result_df, model_mode = "best_overall", use_unet=False):
    
    try:
        out = decode.network_output(config, image, model_mode, use_unet=use_unet)    
        out, predicted_text = decode.decode_one_img_with_info(config, out) 
        lf_points = str([l.tolist() for l in out['lf']])
        result_df.loc[len(result_df.index)] = [image, gt, predicted_text, region_type,
                                          lf_points,
                                          str(out['beginning'].tolist()), str(out['ending'].tolist()),
                                          str(out['sol'].tolist())]
        return result_df
    except RuntimeError as e:
        print('*****get_predictions: Could not process', image, '\n Error', str(e))        

    return None


def get_text_gt(sub_json_file):

    #print('subjson',sub_json_file)
    txt_file = sub_json_file[:-4] + 'gt.txt'
    if os.path.exists(txt_file):
        with open(txt_file) as fin:
            gt = fin.read()           
            if len(gt) != 0:
                gt = gt.split('\n')
                return gt
    print('Not reading from txt')
    with open(sub_json_file) as fin:
        json_gt = json.load(fin)
    gt = []
    for line in json_gt:
        gt.append(line['gt'])
    return gt


def predict_for_one_config(config_file, result_df, 
                           model_mode = "best_overall", region_type="text", test_name="", use_unet=False):
    failed = {'image_file': [], 'gt': []}
    print('config_file', config_file)
    config = get_config(config_file)
    
    if use_unet:
        config['network']['lf']['u_net'] = True
    
    if test_name == "":
        test_file = config['testing']['test_file']
    else:
        test_file = test_name
    print('test:', test_file)
        
    # Read the json test file
    with open(test_file, 'r') as fin:
        json_file_list = json.load(fin)
        

    for [sub_json_file, img_file] in json_file_list:
        # Get the ground truth
        text_gt = get_text_gt(sub_json_file)
        # get the predictions on the images
        # The function below would append the result_df
        print('Image path:', img_file)
        img = cv2.imread(img_file)
        #img = cv2.flip(img, 1)
        #plt.imshow(img)
        #plt.show()
        temp_df = get_predictions(config_file, img_file, text_gt, 
                                    region_type, result_df, model_mode, use_unet=use_unet)
        if temp_df is None:       
            failed['image_file'].append(img_file)
            failed['gt'].append(text_gt)
            
        else:
            result_df = temp_df
        
        torch.cuda.empty_cache()

        
    # Now process all the failed files
    for image_file, gt in zip(failed['image_file'], failed['gt']):
        result_df = split_and_get_predictions(config_file, image_file, result_df, 
                                              region_type, gt, model_mode)
        torch.cuda.empty_cache()
    return result_df
    


def main_function(sets_todo, training_images, main_dir, model_modes, config_name='config', test_name=""):
    
    columns = ['image_file', 'ground_truth','prediction',
               'region_type',
              'lf_points', 'beginning', 'ending', 'SOL']
    for set_dir in sets_todo:
        for total_train in training_images:
            for model_mode in model_modes:
                result_df = pd.DataFrame(columns=columns)
                config_to_use =  main_dir + set_dir + config_name + '_{}.yaml'.format(total_train)
                print('config', config_to_use)
                if test_name != "":
                    test_name_to_use = os.path.join(main_dir, set_dir, test_name)
                else:
                    test_name_to_use = ""
                result_df = predict_for_one_config(config_to_use, result_df, model_mode, test_name=test_name_to_use)
                result_file = main_dir + set_dir + 'result_{}_{}.csv'.format(total_train, model_mode)
                result_df.to_csv(result_file, index=False)
                 
sets_todo = ['set0/', 'set1/', 'set2/']


training_images = [1150]

main_dir = 'trials/public_1100/'
model_modes = ['pretrain' ]
main_function(sets_todo, training_images, main_dir, model_modes)#, test_name="test_list.json")
print('done')