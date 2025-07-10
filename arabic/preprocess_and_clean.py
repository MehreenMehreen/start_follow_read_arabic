#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('coords')
sys.path.append('py3/preprocessing')

import text_gt
import text_cleaning_routines as clean

import line_extraction
import time

import os
import pandas as pd
import json
import cv2
import numpy as np
import traceback
from collections import defaultdict
from scipy import ndimage
from svgpathtools import Path, Line
from scipy.interpolate import griddata
import threading
import points
import text_gt
import argparse


def get_json_gt(filename):
    json_obj = None
    with open(filename) as fin:
        file_content = fin.read()
        json_obj = json.loads(file_content)
    if json_obj is None:
        print('Error reading file', filename)
        
    gt = []
    for key in json_obj.keys():
        if not key.startswith('line_'):
            continue
        
        if "deleted" in json_obj[key] and json_obj[key]["deleted"] == "1":
            if DEBUG:
                print('line deletled', key)
            continue
        gt.append(json_obj[key]["text"])
    return gt

def get_json_poly(filename):
    donotusethis
    json_obj = None
    with open(filename) as fin:
        file_content = fin.read()
        json_obj = json.loads(file_content)
    if json_obj is None:
        print('Error reading file', filename)
        
    all_polygons = []
    all_baselines = []
    for key in json_obj.keys():
        if not key.startswith('line_'):
            continue
        
        if "deleted" in json_obj[key] and json_obj[key]["deleted"] == "1":
            if DEBUG:
                print('line deletled', key)
            continue
        
        poly_pts = json_obj[key]["coord"]
        poly_pts = points.list_to_xy(poly_pts)
        # Check that it is a valid polygon
        if not points.valid_poly(poly_pts):
            if DEBUG:
                print('Invalid polygon in', filename)
            continue
        try:
            baseline = points.get_baseline_chunks(poly_pts)
        except Exception as e:
            if DEBUG:
                print(str(e))
                print(poly_pts)
            continue
        baseline.sort(key=lambda x: x[0], reverse=True)
        all_polygons.append(poly_pts)
        all_baselines.append(baseline)
    return all_polygons, all_baselines
    
def get_json_gt(filename):
    donotusethis
    json_obj = None
    with open(filename) as fin:
        file_content = fin.read()
        json_obj = json.loads(file_content)
    if json_obj is None:
        print('Error reading file', filename)
        
    gt = []
    for key in json_obj.keys():
        if not key.startswith('line_'):
            continue
        
        if "deleted" in json_obj[key] and json_obj[key]["deleted"] == "1":
            if DEBUG:
                print('line deletled', key, filename)
            continue
        gt.append(json_obj[key]["text"])
    return gt

def get_data_from_json(filename):
    json_obj = None
    with open(filename) as fin:
        file_content = fin.read()
        json_obj = json.loads(file_content)
    if json_obj is None:
        print('Error reading file', filename)
        
    all_polygons = []
    all_baselines = []
    all_gt = []
    for key in json_obj.keys():
        if not key.startswith('line_'):
            continue
        
        if "deleted" in json_obj[key] and json_obj[key]["deleted"] == "1":
            if DEBUG:
                print('line deletled', key)
            continue
        
        # Check that gt exists
        if not 'text' in json_obj[key]:
            if DEBUG:
                print('gt empty for key ',k, 'filename', os.path.split(filename)[1])
            continue
        gt = json_obj[key]["text"]
        gt = gt.strip()
        if len(gt) == 0:
            if DEBUG:
                print('gt empty for key ',key, 'filename', os.path.split(filename)[1])
            continue
        # Get lines coordinates
        poly_pts = json_obj[key]["coord"]
        poly_pts = points.list_to_xy(poly_pts)
        # Check that it is a valid polygon
        if not points.valid_poly(poly_pts):
            if DEBUG:
                print('Invalid polygon in', filename)
            continue
        try:
            baseline = points.get_baseline_chunks(poly_pts)
        except Exception as e:
            if DEBUG:
                print(str(e), 'filename', filename)
                print(poly_pts)
            continue
        baseline.sort(key=lambda x: x[0], reverse=True)
        all_polygons.append(poly_pts)
        all_baselines.append(baseline)
        all_gt.append(gt)
    return all_gt, all_polygons, all_baselines


def add_to_df(clean_df, image_file, polygons, baselines, para_number = 1, region_id="main",
                 region_type="text", region_pts=0, line_id=0, ground_truth=""):

    gt = ""
    for ind, (poly, base) in enumerate(zip(polygons, baselines)):
        if len(ground_truth) > 0:
            gt = ground_truth[ind]
        clean_df.loc[len(clean_df)] = [image_file, para_number, region_id, 
                                       region_type, region_pts, line_id, ind+1, 
                                       gt, str(poly), str(base)]
    return clean_df


def get_json_filename(target_dir, img_name):
    for annotator in annotators:
        json_file = img_name[:-4] + f'_annotate_{annotator}.json'
        input_json = os.path.join(target_dir, json_file)
        if os.path.exists(input_json):
            return input_json
    return None

def process_directory(source_dir, columns):
    df = pd.DataFrame(columns=columns)
    para_number = 1
    files = os.listdir(source_dir)
    files.sort()

    for file in files:
        if not file.endswith('.jpg') and not file.endswith('.JPG'):
            continue
        
        input_json = get_json_filename(source_dir, file)
        if input_json is None:
            if DEBUG:
                print('JSON FILE NOT FOUND for file', file, '...skipping')
            continue

        gt_list, polygon_list, baseline_list = get_data_from_json(input_json)
        img_file = os.path.join(source_dir, file)
        df = add_to_df(df, img_file, polygon_list, baseline_list, para_number, ground_truth=gt_list)
        para_number += 1
    return df


    


# In[2]:


def generate_offset_mapping(img, ts, path, offset_1, offset_2, max_min = None, cube_size = None):
    # cube_size = 80

    offset_1_pts = []
    offset_2_pts = []
    # for t in ts:
    for i in range(len(ts)):
        t = ts[i]
        pt = path.point(t)

        norm = None
        if i == 0:
            norm = normal(pt, path.point(ts[i+1]))
            norm = norm / dis(complex(0,0), norm)
        elif i == len(ts)-1:
            norm = normal(path.point(ts[i-1]), pt)
            norm = norm / dis(complex(0,0), norm)
        else:
            norm1 = normal(path.point(ts[i-1]), pt)
            norm1 = norm1 / dis(complex(0,0), norm1)
            norm2 = normal(pt, path.point(ts[i+1]))
            norm2 = norm2 / dis(complex(0,0), norm2)

            norm = (norm1 + norm2)/2
            norm = norm / dis(complex(0,0), norm)

        offset_vector1 = offset_1 * norm
        offset_vector2 = offset_2 * norm

        pt1 = pt + offset_vector1
        pt2 = pt + offset_vector2

        offset_1_pts.append(complexToNpPt(pt1))
        offset_2_pts.append(complexToNpPt(pt2))

    offset_1_pts = np.array(offset_1_pts)
    offset_2_pts = np.array(offset_2_pts)

    h,w = img.shape[:2]

    offset_source2 = np.array([(cube_size*i, 0) for i in range(len(offset_1_pts))], dtype=np.float32)
    offset_source1 = np.array([(cube_size*i, cube_size) for i in range(len(offset_2_pts))], dtype=np.float32)

    offset_source1 = offset_source1[::-1]
    offset_source2 = offset_source2[::-1]

    source = np.concatenate([offset_source1, offset_source2])
    destination = np.concatenate([offset_1_pts, offset_2_pts])

    source = source[:,::-1]
    destination = destination[:,::-1]

    n_w = int(offset_source2[:,0].max())
    n_h = int(cube_size)

    grid_x, grid_y = np.mgrid[0:n_h, 0:n_w]

    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(n_h,n_w)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(n_h,n_w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    rectified_to_warped_x = map_x_32
    rectified_to_warped_y = map_y_32

    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,w)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    warped_to_rectified_x = map_x_32
    warped_to_rectified_y = map_y_32

    return rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min


def dis(pt1, pt2):
    a = (pt1.real - pt2.real)**2
    b = (pt1.imag - pt2.imag)**2
    return np.sqrt(a+b)

def complexToNpPt(pt):
    return np.array([pt.real, pt.imag], dtype=np.float32)

def normal(pt1, pt2):
    dif = pt1 - pt2
    return complex(-dif.imag, dif.real)

def find_t_spacing(path, cube_size):

    l = path.length()
    error = 0.01
    init_step_size = cube_size / l

    last_t = 0
    cur_t = 0
    pts = []
    ts = [0]
    pts.append(complexToNpPt(path.point(cur_t)))
    path_lookup = {}
    for target in np.arange(cube_size, int(l), cube_size):
        step_size = init_step_size
        for i in range(1000):
            cur_length = dis(path.point(last_t), path.point(cur_t))
            if np.abs(cur_length - cube_size) < error:
                break

            step_t = min(cur_t + step_size, 1.0)
            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_t = max(cur_t - step_size, 0.0)
            step_t = max(step_t, last_t)
            step_t = max(step_t, 1.0)

            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_size = step_size / 2.0

        last_t = cur_t

        ts.append(cur_t)
        pts.append(complexToNpPt(path.point(cur_t)))

    pts = np.array(pts)

    return ts

def get_basename(img_name):
    head, tail = os.path.split(img_name)
    basename = 'base_' + tail
    basename = basename[:-4]
    return basename

def handle_single_para(para_df, output_directory, flip=False):
    
    output_data = []
    num_lines = len(para_df)
    if DEBUG:
        print('....Total lines:', num_lines)
    if os.path.exists(para_df.image_file.iloc[0]):
        img = cv2.imread(para_df.image_file.iloc[0])
        if flip:
            img = cv2.flip(img, 1)
    else:
        if DEBUG:
            print('....File not found', para_df.image_file.iloc[0])
        return ''
    basename = get_basename(para_df.image_file.iloc[0])    
    
    all_lines = ""
    
    # get rid of png/jpg extension
    
    for region in [0]:
        region_output_data = []
        #print('in region', region)
        for ind, line in enumerate(para_df.line_number):
            if len(para_df.polygon_pts[ind]) == 0:
                if DEBUG:
                    print('No polygon pts in img', para_df.image_file.iloc[0][-15:],
                          'line number', line)
                continue
            #print('....ind, line', ind, line)
            line_mask = line_extraction.extract_region_mask(img, para_df.polygon_pts[ind])
            masked_img = img.copy()
            masked_img[line_mask==0] = 0

            summed_axis0 = (masked_img.astype(float) / 255).sum(axis=0)
            summed_axis1 = (masked_img.astype(float) / 255).sum(axis=1)

            non_zero_cnt0 = np.count_nonzero(summed_axis0) / float(len(summed_axis0))
            non_zero_cnt1 = np.count_nonzero(summed_axis1) / float(len(summed_axis1))

            avg_height0 = np.median(summed_axis0[summed_axis0 != 0])
            avg_height1 = np.median(summed_axis1[summed_axis1 != 0])

            avg_height = min(avg_height0, avg_height1)
            if non_zero_cnt0 > non_zero_cnt1:
                target_step_size = avg_height0
            else:
                target_step_size = avg_height1

            paths = []
            for i in range(len(para_df.baseline[ind])-1):
                i_1 = i+1

                p1 = para_df.baseline[ind][i]
                p2 = para_df.baseline[ind][i_1]

                p1_c = complex(*p1)
                p2_c = complex(*p2)


                paths.append(Line(p1_c, p2_c))


            if len(paths) == 0:
                if DEBUG:
                    print('Path length is 0', 'for ind', ind)
                continue
            # Add a bit on the end
            tan = paths[-1].unit_tangent(1.0)
            p3_c = p2_c + target_step_size * tan
            paths.append(Line(p2_c, p3_c))

            path = Path(*paths)
            
            try:
                ts = find_t_spacing(path, target_step_size)
                
                #Changing this causes issues in pretraining - not sure why
                target_height = HT

                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, 0, -2*target_step_size, cube_size = target_height)
                warped_above = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(0,0,0))

                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, 2*target_step_size, 0, cube_size = target_height)
                warped_below = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(0,0,0))

                above_scale =  np.max((warped_above.astype(float) / 255).sum(axis=0))
                below_scale = np.max((warped_below.astype(float) / 255).sum(axis=0))


                
                ab_sum = above_scale + below_scale
                above = target_step_size * (above_scale/ab_sum)
                below = target_step_size * (below_scale/ab_sum)

                above = target_step_size * (above_scale/(target_height/2.0))
                below = target_step_size * (below_scale/(target_height/2.0))
                target_step_size = above + below
                ts = find_t_spacing(path, target_step_size)
                if len(ts) <= 1:
                    if DEBUG:
                        print('Not doing line', line)
                    continue
                rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(masked_img, ts, path, below, -above, cube_size=target_height)

                rectified_to_warped_x = rectified_to_warped_x[::-1,::-1]
                rectified_to_warped_y = rectified_to_warped_y[::-1,::-1]

                warped_to_rectified_x = warped_to_rectified_x[::-1,::-1]
                warped_to_rectified_y = warped_to_rectified_y[::-1,::-1]

                warped = cv2.remap(img, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC, borderValue=(255,255,255))
            except:
                if DEBUG:
                    print('Not doing line', line)
                continue
            
            
            mapping = np.stack([rectified_to_warped_y, rectified_to_warped_x], axis=2)

            top_left = mapping[0,0,:] / np.array(img.shape[:2]).astype(np.float32)
            btm_right = mapping[min(mapping.shape[0]-1, target_height-1), min(mapping.shape[1]-1, target_height-1),:] / np.array(img.shape[:2]).astype(np.float32)


            line_points = []
            for i in range(0,mapping.shape[1],target_height):

                x0 = float(rectified_to_warped_x[0,i])
                x1 = float(rectified_to_warped_x[-1,i])

                y0 = float(rectified_to_warped_y[0,i])
                y1 = float(rectified_to_warped_y[-1,i])

                line_points.append({
                    "x0": x0, 
                    "x1": x1, 
                    "y0": y0, 
                    "y1": y1, 
                })
                
                
                                
            ###Mehreen add for viewing

#            plt.imshow(img) # or display line warped
#            print("****", line_points)
#            for coord in line_points:
#                x = coord["x0"]
#                y = coord["y0"]
#                x1 = coord["x1"]
#                y1 = coord["y1"]
                #rect = patches.Rectangle((x, y), np.abs(x-coord[2]), np.abs(y-coord[3]), facecolor='green')
#                rect = patches.Rectangle((x, y), 10, 10, facecolor='blue')
#                rect1 = patches.Rectangle((x1, y1), 10, 10, facecolor='red')
#                plt.gca().add_patch(rect)  
#                plt.gca().add_patch(rect1)
#            rect0 = patches.Rectangle((line_points[0]["x0"], line_points[0]["y0"]), 10, 10, facecolor='yellow') 
#            plt.gca().add_patch(rect0)
#            plt.show()
             ## ENd mehreen add for view   
                
            
            output_file = os.path.join(output_directory, 
                          basename, "{}~{}~{}.png".format(basename, region, line))
            warp_output_file = os.path.join(output_directory, basename, "{}-{}.png".format(basename, line))
            warp_output_file_save = os.path.join(basename, "{}-{}.png".format(basename, str(len(region_output_data))))
            save_file = os.path.join(basename, "{}~{}~{}.png".format(basename, region, line))
            region_output_data.append({
                "gt": para_df.ground_truth[ind],
                "image_path": save_file,
                "sol": line_points[0],
                "lf": line_points,
                "hw_path": warp_output_file #MEhreen commentwarp_output_file_save
            })
            #print('****', output_file)
            if not os.path.exists(os.path.dirname(output_file)):
                try:
                    os.makedirs(os.path.dirname(output_file))
                except OSError as exc:
                    raise Exception("Could not write file")

            cv2.imwrite(warp_output_file, warped)

        output_data.extend(region_output_data)

    
    if len(region_output_data) == 0:
        return ''
        
    output_data_path =os.path.join(output_directory, basename, "{}.json".format(basename))
    if not os.path.exists(os.path.dirname(output_data_path)):
        os.makedirs(os.path.dirname(output_data_path))

    with open(output_data_path, 'w') as f:
        json.dump(output_data, f)

    return output_data_path    
    
def convert_str_to_int_tuples(df_col):
    tmp_col = []
    for ind, item in enumerate(df_col):
        if not pd.isna(item):
            item = eval(item)    
            tmp_col.append([(round(x[0]), round(x[1])) for x in item])
        else:
            tmp_col.append([])
    return tmp_col

def rotate_polygon(p):
    if len(p) == 8 or len(p) == 7:
        poly = p[4:]
        poly.extend(p[0:4])
        return poly
    if len(p) == 4:
        poly = [p[2], p[3], p[0], p[1]]
        return poly
    else:
        print("something wrong", p)
        return []
    
def rotate_poly_list(df_col):
    poly_list = [rotate_polygon(p) for p in df_col]
    return poly_list


def rotate_baseline_list(df_col):
    b_list = [b[::-1] for b in df_col]
    return b_list
    
# Reverse is true for right to left reading order    
def remove_duplicate_baseline(baseline, reverse):
    baseline.sort(key=lambda x: x[0], reverse=reverse)
    unique_pts = [baseline[0]]
    for pt in baseline[1:]:
        if pt != unique_pts[-1]:
            unique_pts.append(pt)
    
    return unique_pts

def get_img_dim(filename):
    img = cv2.imread(filename)
    ht, width = img.shape[:2]
    return (ht, width)
# Subtract all x-coord from img_width as for Arabic we flip the image horizontally 
# Right to left reading order
# Set reverse=True for baseline so that reading order becomes left to righ
def flip_x_coord(df_col, img_width, reverse=False):
    new_col = []
    for ind, item in enumerate(df_col):
        flipped = [(img_width-x, y) for (x,y) in item]
        if reverse:
            flipped = flipped[::-1]
        new_col.append(flipped)
    return new_col

def split_dataframe(df, total_splits):
    if total_splits == 1:
        return [df]
    
    images = [i for i in df.image_file]
    images = list(set(images))
    images.sort()

    split_size = int(np.ceil(len(images)/total_splits))
    to_split = [0] * total_splits
    df_array = [0] * total_splits
    for i in range(total_splits):
        end_ind = min((i+1)*split_size, len(images))
        to_split[i] = images[i*split_size: end_ind]

        df_array[i] = df[df['image_file'].isin(to_split[i])].copy().reset_index(drop=True)
    return df_array



def run_preprocess_thread(df, out_path, set_name, region_types, verbose):
    files = df.image_file
    
    for para_numb in set(df.paragraph_number):
        
        para_df = df[df.paragraph_number == para_numb]
        para_df = para_df.copy()
        para_df = para_df.reset_index(drop=True)
        _, image_width = get_img_dim(para_df.image_file[0])
        
        # To have a left to right reading order
        para_df.baseline = flip_x_coord(para_df.baseline, image_width, reverse=False)
        para_df.baseline = [remove_duplicate_baseline(b, reverse=False) for b in para_df.baseline]
        
        para_df.polygon_pts = flip_x_coord(para_df.polygon_pts, image_width) 
        

        valid_region = False
        if not pd.isna(para_df['region_type'][0]):
            valid_region = any(s in para_df['region_type'][0] for s in region_types)
        if not valid_region:
            continue
        
        img_path = para_df.image_file.iloc[0]
        basename = get_basename(img_path)
        
        tmp_path = os.path.join(out_path, set_name, basename, basename + '.json')
        
        if os.path.isfile(tmp_path):
            json_path = tmp_path
            if verbose:
                print('... Done', basename)
        else: 
            if verbose:
                print('....Doing', basename)
            json_path = handle_single_para(para_df, out_path+set_name, flip=True)    
            
        # para not added    
        if len(json_path) == 0:
            if verbose:
                print('Not added for', basename)
            continue

# Do not process a file again. Only process the files that are not done
def process_notdone_arabic_dir(out_path, set_name, region_types, input_file=None, 
                               total_threads=1, verbose=False):
    
    if input_file is None:
        data_df = pd.read_csv(os.path.join(out_path, set_name +'.csv'))
    else:
        data_df = pd.read_csv(input_file)
    
    data_df.baseline = convert_str_to_int_tuples(data_df.baseline)
    data_df.polygon_pts = convert_str_to_int_tuples(data_df.polygon_pts)
    df_array = split_dataframe(data_df, total_threads)
    threads = []
    for df in df_array:
        thread = threading.Thread(target=run_preprocess_thread, 
                              args=(df, out_path, set_name, region_types, verbose))
        thread.start()
        threads.append(thread)
    
    # Wait for finish
    for thread in threads:
        thread.join()
    
    return create_final_json(out_path, set_name, region_types, 
                      input_file=input_file)

# This function can be run if multiple threads cannot be run  
# Also, run this in the end to make sure all files are processed
def create_final_json(out_path, set_name, region_types, input_file=None, verbose=True):    
    if input_file is None:
        df = pd.read_csv(os.path.join(out_path, set_name +'.csv'))
    else:
        df = pd.read_csv(input_file)
    
    
    df.baseline = convert_str_to_int_tuples(df.baseline)
    df.polygon_pts = convert_str_to_int_tuples(df.polygon_pts)
    
    files = df.image_file
    if verbose:
        print(f'Total files in set {set_name}: {len(set(files))}')
    
    all_ground_truth = []
    for para_numb in set(df.paragraph_number):
        
        para_df = df[df.paragraph_number == para_numb].copy().reset_index(drop=True)
        _, image_width = get_img_dim(para_df.image_file[0])
        
        # To have a left to right reading order
        para_df.baseline = flip_x_coord(para_df.baseline, image_width, reverse=False)
        para_df.baseline = [remove_duplicate_baseline(b, reverse=False) for b in para_df.baseline]
        
        para_df.polygon_pts = flip_x_coord(para_df.polygon_pts, image_width) 
        

        valid_region = False
        if not pd.isna(para_df['region_type'][0]):
            valid_region = any(s in para_df['region_type'][0] for s in region_types)
        if not valid_region:
            continue
        
        img_path = para_df.image_file.iloc[0]
        basename = get_basename(img_path)
        
        tmp_path = os.path.join(out_path, set_name, basename, basename + '.json')
        
        
        if os.path.isfile(tmp_path):
            json_path = tmp_path
            if verbose:
                print('... Done', basename)
        else: 
            if verbose:
                print('....Doing', basename)
            json_path = handle_single_para(para_df, out_path+set_name, flip=True)  
            
        # para not added    
        if len(json_path) == 0:
            if verbose:
                print('Not added', basename)
            continue
            
        all_ground_truth.append([json_path, img_path])
        if len(all_ground_truth)%100 == 0:
            if verbose:
                print('done', len(all_ground_truth))
        
    return all_ground_truth
    
        
    

def add_visual_order_of_text(json_list_file):

    with open(json_list_file) as fin:
        file_list = json.load(fin)
    
    
    
    for [json_file, img_file] in file_list:
        total = 0
        changed = 0

        # Corresponds to one image file
        with open(json_file) as fin:
            json_list = json.load(fin)

        for ind, json_gt in enumerate(json_list):    
            if 'clean_visual_order' in json_gt:
                gt = json_gt['original_gt']
            else:
                gt = json_gt['gt']
            old_gt = gt
            if type(gt) == float:
                gt = ""
            gt = gt.strip()
            if len(gt) == 0 or gt == 'None':
                gt = ""
            # GEt rid of consec spaces
            gt = ' '.join(gt.split())
            clean_visual_order = clean.get_clean_visual_order(gt)
            if clean_visual_order != gt:  
                
                changed += 1
            total += 1 
            json_list[ind]['original_gt'] = old_gt
            json_list[ind]['clean_visual_order'] = clean_visual_order
            json_list[ind]['gt'] = clean_visual_order


        with open(json_file, 'w') as fout:
            json.dump(json_list, fout)

        print('main file', json_file, 'Changed', changed, 'total', total)

    
    
def create_visual_logical_ordering_text_files(json_list_file):

    # Write the clean visual order
    with open(json_list_file) as fin:
        file_list = json.load(fin)
    
    for [json_file, img_file] in file_list:
        original_txt_file = json_file[:-4] + 'gt.txt'
        logical_txt_file = json_file[:-4] + 'logical_gt.txt'
        if os.path.exists(original_txt_file):
            with open(original_txt_file) as fin:
                gt = fin.read()
        else:
            gt = ""
        clean_visual_text = []
        logical_text = []
        with open(json_file) as fin:
            json_list = json.load(fin)
        
        sorted_json = sorted(json_list, 
                             key=lambda item: (item["sol"]["y0"], -item["sol"]["x0"]))
        for obj in sorted_json:
            if type(obj['clean_visual_order']) == str:
                clean_visual_text.append(obj['clean_visual_order'] )
            if type(obj['original_gt']) == str:
                logical_text.append(obj['original_gt'] )
                
                
            #if obj['clean_visual_order'] != obj['original_gt']:
            #    print('*************', obj['clean_visual_order'],  obj['original_gt'])
                
                
        clean_visual_text = '\n'.join(clean_visual_text)
        logical_text = '\n'.join(logical_text)
        
        with open(original_txt_file, 'w') as fout:
            fout.write(clean_visual_text)
          
        with open(logical_txt_file, 'w') as fout:
            fout.write(logical_text)   

   
    
def preprocess_for_arabic(csv_path, set_names):

    for set_to_use in set_names:
        json_file = os.path.join(csv_path, set_to_use + '.json')
        #if os.path.exists(json_file):
        #    if DEBUG:
        #        print('JSON file exists, skipping', set_to_use + '.json')
        #    continue
        train_gt = process_notdone_arabic_dir(csv_path, set_to_use, 
                                              ['paragraph', 'text'], verbose=DEBUG,
                                              total_threads=5
                                              )
     
        
    
    
        with open(json_file, 'w') as f:
            json.dump(train_gt, f)
            
    
        # Correct the ground truths...need to create visual order of text while keeping logical order
        add_visual_order_of_text(json_file)

        # Create visual and logical ordering text files containing only text of the image
        create_visual_logical_ordering_text_files(json_file)
        
        


def create_flipped_images(output_img_dir, directory, set_names, suffix="sfr"):

    # These are hard coded
    input_json_files = [f'{s}.json' for s in set_names]
    output_json_files = [f'{s}_{suffix}.json' for s in set_names]

    
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
        
    for input_file, output_file, set_name in zip(input_json_files, 
                                                 output_json_files,
                                                 set_names):
        out_json_filename = os.path.join(directory, output_file)    
        #if os.path.exists(out_json_filename):
        #    if DEBUG:
        #        print('JSON exists, skipping', out_json_filename)
        #    continue

        output_dir = os.path.join(output_img_dir, set_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        input_file = os.path.join(directory, input_file)
        with open(input_file, 'r') as fin:
            json_list = json.load(fin)
        print('Total input files to do for flipped', len(json_list))
        output_json = []
        for [json_path, img_path] in json_list:
            _, img_filename = os.path.split(img_path)
            # Put the image in the directory with set name
            output_img_filename = os.path.join(output_dir, img_filename)
            output_json.append([json_path, output_img_filename])
            if os.path.exists(output_img_filename):
                print('Already exists', output_img_filename)                
                continue
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            
            cv2.imwrite(output_img_filename, img)
            
        
        with open(out_json_filename, 'w') as fout:
            json.dump(output_json, fout)


    
# To define command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Preprocess all files")
    # Define a list argument
    # Main input directory with preprocessed files
    parser.add_argument('--src_directory', type=str, default='data_files/',
                        help='Main input directory of img+json files, default is data_files/')
    parser.add_argument('--set_names', type=str, default=['public', 'restricted'], nargs='+',
                        help='List of directories, default directories are public and restricted')
    parser.add_argument('--annotators', type=str, default=['carlos', 'georges', 'carlosai', 'georgesai', 'elias', ''], nargs='+',
                        help='Annotator names, can be blank. Is there for consistency with scribeArabic software')
    

    args = parser.parse_args()
    return args
    


if __name__ == "__main__":

    args = get_args()
    src_directory = args.src_directory
    DEBUG = True
    HT = 60

    src_directory = args.src_directory    
    set_names = args.set_names
    annotators = args.annotators
   
    COLUMNS = ['image_file', 'paragraph_number', 
                      'region_id', 'region_type', 'region_pts',
                      'line_id', 'line_number', 
                      'ground_truth', 'polygon_pts', 'baseline']

    
    sfr_arabic_dir = os.path.join(src_directory, 'sfr_arabic/')
    # ----------- Step 1
    print('Creating CSV files, please wait ...')
    for s in set_names:
        target_csv = os.path.join(sfr_arabic_dir, f'{s}.csv')
        #if os.path.exists(target_csv):
        #    continue    
        df = process_directory(os.path.join(src_directory, s), COLUMNS)
        df.to_csv(target_csv, index=False)

    print('CSV files Created')

    # -------------Step 2
    start_time = time.time()
    print('Creating SFR data...')
    preprocess_for_arabic(sfr_arabic_dir, set_names)
    end_time = time.time()
    print('Time: ', end_time - start_time)
    print('Intermediate JSON created')
    

    # -------------Step 3
    print('Creating flipped images ...')
    flipped_dir = os.path.join(sfr_arabic_dir, 'flipped_images/')
    create_flipped_images(flipped_dir, sfr_arabic_dir, set_names)
    


    
    
    print('All preprocessing is done')
        

# Example run: python arabic/preprocessing_and_clean.py --src_directory ~/mehreen/datasets/scribeArabic_2/ --set_names Nasrallah --annotators sfr lp_sfr







