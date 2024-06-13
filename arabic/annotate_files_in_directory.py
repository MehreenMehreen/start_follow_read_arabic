import sys
sys.path.append('arabic/')
sys.path.append('py3/')
sys.path.append('coords')
import test_hw_helper_routines as hw
import json
import os
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import pandas as pd
import numpy as np
import decode_one_image as decode
import post_process_routines as post
import points

HW = None
def load_network(pt_file = 'hw.pt', suffix="", device="cuda"):
    global HW
    config = hw.get_config(config_file)    
    #print("config is ", config)
    idx_to_char = hw.load_char_set(config['network']['hw']['char_set_path'])
    pt_filename = os.path.join(config['pretraining']['snapshot_path'], pt_file)
    print('...Using snapshot', pt_filename)
    
    if len(suffix) > 0:
        pt_filename = pt_filename[:-3] + suffix + '.pt'
    HW = hw.load_HW(config['network']['hw'], pt_filename)
    device = torch.device(device)
    HW.to(device)
    HW.eval()
    return idx_to_char
    
# The case when lines are already annotated by hand. Will only OCR line images    
def get_annotated_lines_prediction():
    idx_to_char = load_network()    
    with open(input_json) as fin:
        lines_json = json.load(fin)

    with open(annotation_json) as fin:
        annotations = json.load(fin)

    for ind, line in enumerate(lines_json):
        print(line['hw_path'])
        img_path = line['hw_path']
        img = cv2.imread(img_path)
        img = cv2.flip(img, 1)
        plt.imshow(img)
        prediction = hw.get_predicted_str(HW, line['hw_path'], idx_to_char, device="cuda", flip=False)
        plt.show()
        print(prediction)
        annotations['line_'+str(ind+1)]['text'] = prediction

        

def predict_annotations_for_directory(input_dir, config_file, annotator, model_mode="pretrain", skip_if_json_exists=False):
    files = os.listdir(input_dir)
    files.sort()
    for f in files:        
        if not f.lower().endswith('.jpg'):
            continue
        img_file = os.path.join(input_dir, f)
        json_file = img_file[:-4] + '_annotate_' + annotator + '.json'
        if os.path.exists(json_file) and skip_if_json_exists:
            print('already done', json_file)
            continue
        print('doing', img_file)
        out = decode.network_output(config_file, img_file, flip=True, model_mode=model_mode)  
        out, predicted_text = decode.decode_one_img_with_info(config_file, out, flip=True) 
        poly_list = post.get_polygon_list_tuples(out)
        
        # Get rid of degenerate points
        to_del_ind = []
        for ind, p in enumerate(poly_list):
            if len(p) < 3:
                to_del_ind.append(ind)
        if len(to_del_ind) > 0:
            print('Deleting poly at index', to_del_ind)
            poly_list = [poly_list[i] for i in range(len(poly_list)) if i not in to_del_ind]
            predicted_text = [predicted_text[i] for i in range(len(predicted_text)) if i not in to_del_ind]
        
        del_list, poly_list = post.get_poly_no_overlap(img_file, poly_list, 0.7)
        
        if len(del_list) > 0:
            print('polygons deleted', len(del_list), del_list)
            print(len(poly_list))
        
        predicted_text = [predicted_text[i] for i in range(len(predicted_text)) if i not in del_list]
        poly_list = post.flip_polygon(img_file, poly_list)
        #post.draw_image_with_poly("", img_file, poly_list, convert=False)
        page_json = post.create_annotations_json(predicted_text, poly_list)
        
        with open(json_file, 'w') as fout:
            json.dump(page_json, fout)


if __name__ == "__main__":   
    if len(sys.argv) < 3:
        error_msg = 'Create annotation JSON files for a page image. They can be viewed using ScribeArabic software'
        error_msg += 'Specify input directory with images as first argument and config file as second argument\n'
        error_msg += 'Optionally specify whether to overwrite JSON file if it already exists (default is 0), 1 means do not overwrite'
        print(error_msg)
        sys.exit()
        
    annotator = 'sfr'
    input_dir = sys.argv[1]
    config_file = sys.argv[2]
    skip_if_json_exists = False
    if len(sys.argv) > 3:
        skip_if_json_exists = int(sys.argv[3]) == 1
            
            
    predict_annotations_for_directory(input_dir, config_file, annotator, model_mode="pretrain",
                                      skip_if_json_exists=skip_if_json_exists)
    print('done')
        
