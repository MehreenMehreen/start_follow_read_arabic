# This script will write the results csv files in all folders
import sys
sys.path.append('py3/')
sys.path.append('arabic/')
sys.path.append('coords/')
import json
import pandas as pd
import yaml
import decode_one_image as decode
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import post_process_routines as post
from utils import error_rates

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
    

# This function will create a results_{total_train}_{model_mode}.csv file to contain results of SOL, LF, HW
def main_function(sets_todo, training_images, main_dir, model_modes=['pretrain'], 
                  config_name='config', test_name=""):
    
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


#CER AFTER REMOVING OVERLAPPING POLYGONS
# Will compute 

def compute_cer(main_dir, sets_todo, total_train, model_mode='pretrain', no_diacritics=False):
    columns = ['image_path', 'CER', 'CER_corrected', 'WER', 'WER_corrected']
    set_cer = []
    set_cer_corrected = []
    set_wer = []
    set_wer_corrected = []

    for s in sets_todo:
        cer_df = pd.DataFrame(columns=columns)
        total = 0
        result_csv = main_dir + s + f'result_{total_train}_{model_mode}.csv'
        df = pd.read_csv(result_csv)
        sum_cer = 0
        sum_corrected_cer = 0
        sum_wer = 0
        sum_corrected_wer = 0
        for ind in range(len(df)):
            out = {'image_path': df.image_file[ind], 'lf': eval(df.lf_points[ind]), 
                   'beginning': eval(df['beginning'][ind]),
                   'ending': eval(df['ending'][ind])}
            p = post.get_polygon_list_tuples(out)
            del_list, poly_non_overlapping = post.get_poly_no_overlap(df.image_file[ind], p, 0.7)
            if type(df.ground_truth[ind]) == float:
                print('skipping', df.image_file[ind])
                continue
            gt = df.ground_truth[ind]
            ground_truth = gt.replace('nan', "''")
            ground_truth = eval(ground_truth)

            if len(ground_truth) == 0:
                print('skipping', df.image_file[ind])
                continue
            #print(ground_truth)
            prediction = eval(df.prediction[ind])
            cer = error_rates.cer('\n'.join(ground_truth), '\n'.join(prediction))
            cer_corrected = cer
            sum_cer += cer
            wer = error_rates.wer('\n'.join(ground_truth), '\n'.join(prediction))
            wer_corrected = wer
            sum_wer += wer
            if len(del_list) > 0:  
                prediction = [prediction[i] for i in range(len(prediction)) if i not in del_list]
                cer_corrected = error_rates.cer('\n'.join(ground_truth), '\n'.join(prediction))
                wer_corrected = error_rates.wer('\n'.join(ground_truth), '\n'.join(prediction))
                sum_corrected_cer += cer_corrected
                sum_corrected_wer += wer_corrected
            else:
                sum_corrected_cer += cer
                sum_corrected_wer += wer
            total += 1
            cer_df.loc[len(cer_df)] = [df.image_file[ind], cer, cer_corrected, wer, wer_corrected]
            print('Image: ', df.image_file[ind])
            print('CER', cer, 'CER_corrected', cer_corrected, 'WER', wer, 'WER_corrected', wer_corrected)      
            
        cer_csv = main_dir + s + f'cer_wer_result_{total_train}_{model_mode}.csv'    
        cer_df.to_csv(cer_csv)
        set_cer.append(sum_cer/total)
        set_cer_corrected.append(sum_corrected_cer/total)
        set_wer.append(sum_wer/total)
        set_wer_corrected.append(sum_corrected_wer/total)
        
    return set_cer, set_cer_corrected, set_wer, set_wer_corrected

               
                
# To define command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Get CER and WER of entire page image")
    # Define a list argument
    # Main input directory with preprocessed files
    parser.add_argument('--main_dir', type=str, default= 'trials/public_1100/',
                        help='Main trial directory, default is trials/public_1100/')
    parser.add_argument('--train_valid', type=int, default= 1150, 
                        help='Total training plus validation images, default for public set is 1150')
    parser.add_argument('--total_sets', type=int, default= 3,
                        help='Total sets in trial directory. Default is 3 and will process set0, set1, set2')
    

    args = parser.parse_args()
    return args
                    
if __name__ == "__main__":
        
    args = get_args()
    total_sets = args.total_sets
    # Names of sets set0, set1, set2, ...
    sets_todo = [f'set{n}/' for n in range(total_sets)]
    training_images = [args.train_valid]
    main_dir = args.main_dir
    if main_dir[-1] != '/':
        main_dir += '/'
    # Make the intermediate csv result file
    main_function(sets_todo, training_images, main_dir)#, test_name="test_list.json")
    # Compute CER + WER (raw) and after removing overlapping polygons
    set_cer, set_cer_corrected, set_wer, set_wer_corrected = compute_cer(main_dir, sets_todo, args.train_valid)
                                                                     

    print('****SUMMARY')
    print('CER: ', set_cer)
    print('CER corrected: ', set_cer_corrected)
    print('WER: ', set_wer)
    print('WER corrected: ', set_wer_corrected)
    print('Mean cer without correction: {:.3f}'.format(np.mean(set_cer)))
    print('Mean cer with correction: {:.3f}'.format(np.mean(set_cer_corrected)))

    print('Mean wer without correction: {:.3f}'.format(np.mean(set_wer)))
    print('Mean wer with correction: {:.3f}'.format(np.mean(set_wer_corrected)))
   
    print('done')