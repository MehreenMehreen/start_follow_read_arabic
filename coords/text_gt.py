# Extract text from json file
# As all lines need sorting etc. this is being added to coords folder

import sys
import points
import json
import pandas as pd
import os
import numpy as np


def sort_lines(lines_list):
    
    line_starts = [line['baseline'][0] for line in lines_list]
    sorted_starts = sorted(enumerate(line_starts), key=lambda x: (x[1][1], -x[1][0]))
    sorted_start_ind = [x[0] for x in sorted_starts]
    sorted_lines = [lines_list[i] for i in sorted_start_ind]
    
    return sorted_lines

def is_valid_key(key, json_obj):
    if not key.lower().startswith('line_'):
        return False
    if "deleted" in json_obj[key].keys() and json_obj[key]["deleted"] != "0":
        return False
    # No text field
    if not 'text' in json_obj[key]:
        return False
    # Text empty
    json_obj[key]['text'] = json_obj[key]['text'].replace('\t', ' ')
    if json_obj[key]['text'].strip() == "":
        return False
    return True

def get_text(json_file, return_list=False):
    with open(json_file) as fin:
        json_obj = json.load(fin)
    to_remove_ind = []
    # Get keys in json_obj
    keys = list(json_obj.keys())
    # Get list of line objects
    lines = [json_obj[k] for k in keys if is_valid_key(k, json_obj)]
    # Get baseline of each line
    for ind, line in enumerate(lines):
        poly_pts = line["coord"]
        poly_pts = points.list_to_xy(poly_pts)
        if len(poly_pts) <= 2:
            print(json_file, len(poly_pts))
            to_remove_ind.append(ind)
        if not points.valid_poly(poly_pts):
            to_remove_ind.append(ind)
            continue
        try:
            baseline = points.get_baseline_chunks(poly_pts)
            baseline.sort(key=lambda x: x[0], reverse=True)
            line['baseline'] = baseline
        except Exception as e:
            #print(len(poly_pts))
            #print(poly_pts)
            #print(json_file)
            to_remove_ind.append(ind)

    # REmove the lines causing exception
    cleaned_lines = [lines[ind] for ind in range(len(lines)) if not ind in to_remove_ind]
    # Sort the lines
    sorted_lines = sort_lines(cleaned_lines)
    
    text = []
    for l in sorted_lines:
        text.append(l["text"])
    if return_list:
        return text
    return '\n'.join(text)
            
def get_json_file(img_fullname):
    dir, img_name = os.path.split(img_fullname)
    json_files = []
    #annotators = []
    base_file = img_name[:-4]
    files = os.listdir(dir)
    for f in files:
        prefix = base_file + '_annotate_'
        if f.startswith(prefix):
            
            # Check if its a timestamp in filename
            partial_string = f[len(prefix):]
            ind1 = partial_string.rfind('.')
            ind2 = partial_string.find('.')
            
            
            if (ind1 == ind2):
                json_files.append(f)
                
    if len(json_files) > 1:
        print('More than one json found...returning 0th one', json_files)
    if len(json_files) == 0:
        print('No json found')
        return None
    
    return os.path.join(dir, json_files[0])

