import os
import sys
import torch

from utils.continuous_state import init_model
from e2e import e2e_model, e2e_postprocessing, visualization
from e2e.e2e_model import E2EModel

import torch
from torch import nn
from torch.autograd import Variable

import json
import cv2
import numpy as np

import codecs
import yaml

from hw import grid_distortion
from collections import defaultdict
import operator
import pandas as pd
from utils import error_rates
import matplotlib.pyplot as plt
import argparse


# Network output on one image
# Will read from file if org_img is none
def network_output(config_file, image_path, model_mode = "best_overall", flip=False, use_unet=False, org_img=None):
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)    
        
    if use_unet:
        config['network']['lf']['u_net'] = True
        #print('config changed')
    
    char_set_path = config['network']['hw']['char_set_path']
    ### Change hw's num_of_outputs in config
    with open(char_set_path) as f:
        char_set = json.load(f)
    
    config["network"]["hw"]["num_of_outputs"] = len(char_set['idx_to_char']) + 1
    
    
    sol, lf, hw = init_model(config, sol_dir=model_mode, lf_dir=model_mode, hw_dir=model_mode)

    e2e = E2EModel(sol, lf, hw)
    dtype = torch.cuda.FloatTensor
    e2e.eval()
    
    if org_img is None:
        org_img = cv2.imread(image_path)
    if flip:
        org_img = cv2.flip(org_img, 1)

    target_dim1 = 512
    s = target_dim1 / float(org_img.shape[1])

    pad_amount = 128
    org_img = np.pad(org_img, ((pad_amount,pad_amount),(pad_amount,pad_amount), (0,0)), 'constant', constant_values=255)
    before_padding = org_img

    target_dim0 = int(org_img.shape[0] * s)
    target_dim1 = int(org_img.shape[1] * s)

    full_img = org_img.astype(np.float32)
    full_img = full_img.transpose([2,1,0])[None,...]
    full_img = torch.from_numpy(full_img)
    full_img = full_img / 128 - 1


    img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img.transpose([2,1,0])[None,...]
    img = torch.from_numpy(img)
    img = img / 128 - 1

    out = e2e.forward({
        "resized_img": img,
        "full_img": full_img,
        "resize_scale": 1.0/s
    }, use_full_img=True)

    out = e2e_postprocessing.results_to_numpy(out)

    if out is None:
        print ("No Results")
        return

    # take into account the padding
    out['sol'][:,:2] = out['sol'][:,:2] - pad_amount
    for l in out['lf']:
        l[:,:2,:2] = l[:,:2,:2] - pad_amount

    out['image_path'] = image_path

    return out


def decode_one_img_with_info(config_path, out, visualize=False, flip=False, org_img=None):
       
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)    

    char_set_path = config['network']['hw']['char_set_path']
    
    

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    out = dict(out)

    image_path = str(out['image_path'])
    #print(image_path)
    if org_img is None:
        org_img = cv2.imread(image_path)
    if flip:
        org_img = cv2.flip(org_img, 1)

    # Postprocessing Steps
    out['idx'] = np.arange(out['sol'].shape[0])
    out = e2e_postprocessing.trim_ends(out)
    e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
    out = e2e_postprocessing.postprocess(out,
        sol_threshold=config['post_processing']['sol_threshold'],
        lf_nms_params={
            "overlap_range": config['post_processing']['lf_nms_range'],
            "overlap_threshold": config['post_processing']['lf_nms_threshold']
        }
    )
    order = e2e_postprocessing.read_order(out)
    e2e_postprocessing.filter_on_pick(out, order)    
        
    # Get output strings and CER    
    output_strings = []
    output_strings, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)    
    return out, output_strings

def write_line_images(images, parent_img_fullpath, result_dir='Result', flip=True):
    directory = os.path.dirname(parent_img_fullpath)
    parent_basename = os.path.basename(parent_img_fullpath)
    dir_basename = parent_basename[0:parent_basename.rfind('_')]
    result_dir = os.path.join(directory, result_dir)
    new_directory = os.path.join(directory, result_dir, dir_basename)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    # Get rid of file extension
    parent_basename = parent_basename[0:parent_basename.rfind('.')]
    for ind, img in enumerate(images):
        
        filename = os.path.join(new_directory, 
                                parent_basename + '_' + str(ind) + '.png')
       
        if flip:
            img = cv2.flip(img, 1)
        cv2.imwrite(filename, img)


# Write a one time header if needed
def write_empty(result_file):
    result_df = pd.DataFrame(columns = ['image_file', 'ground_truth','prediction',
                                        'region_type',
  #                                      'gt_poly',
                                        'lf_points', 'beginning', 'ending', 'SOL'])
    result_df.to_csv(result_file, index=False)

    
# TODO: Verify that (x,y) are the first two of SOL output
def add_offset_to_sol(sol, offset):
    for ind in range(len(sol)):
        sol[ind][0] += offset[0]
        sol[ind][1] += offset[1]
    return sol
        
        
def add_offset_to_lf(lf, offset):
    for pt_ind in range(len(lf)):
        for line_ind in range(len(lf[0])):
            lf[pt_ind][line_ind][0][0] = lf[pt_ind][line_ind][0][0] + offset[0]
            lf[pt_ind][line_ind][1][0] = lf[pt_ind][line_ind][1][0] + offset[1]
            lf[pt_ind][line_ind][0][1] = lf[pt_ind][line_ind][0][1] + offset[0]
            lf[pt_ind][line_ind][1][1] = lf[pt_ind][line_ind][1][1] + offset[1]
    return lf

# The merged output will be in out1
def merge_out(out1, out2, offset):
    lf1 = out1['lf']
    lf2 = add_offset_to_lf(out2['lf'], offset)
    out1['lf'].extend(lf2)
    
    out1['beginning'] = np.concatenate((out1['beginning'], out2['beginning']))
    out1['ending'] = np.concatenate((out1['ending'], out2['ending']))
    
    sol2 = add_offset_to_sol(out2['sol'], offset)
    out1['sol'] = np.vstack((out1['sol'], sol2))
    return out1

def split_image_horizontal(img_file):
    f1 = img_file[:-4] + '_1' + '.jpg'
    f2 = img_file[:-4] + '_2' + '.jpg'
    img = cv2.imread(img_file)
    [ht, width, colors] = img.shape
    ht1 = int(ht/2)
    img1 = img[:ht1, :, :]
    img2 = img[ht1:, :, :]
    cv2.imwrite(f1, img1)
    cv2.imwrite(f2, img2)
    return f1, f2, ht1

        
