import numpy as np
import cv2
import json
import sys
sys.path.append('../../')
from coords import points
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

from utils import error_rates
from arabic_reshaper import ArabicReshaper

# This file has routines for working with CERs and polygons

def correct_pt(value, max_value):
    boundary = False
    if value < 0:
        value = 0
        boundary = True
    if value >= max_value:
        value = max_value - 1
        boundary = True
    return [value, boundary]

# Each polygon is a list of (x,y) tuples
def get_polygon_list_tuples(out):
    img = cv2.imread(out["image_path"])
    img_height, img_width = img.shape[:2]
    polygon_list = []
    prev = [-1, -1]
    for line_ind in range(len(out['lf'][0])):
        polygon = []
        begin_ind = out['beginning'][line_ind]
        end_ind = out['ending'][line_ind]
        begin_ind = int(np.floor(begin_ind))
        end_ind = int(np.ceil(end_ind))
        end_ind = min(end_ind, len(out['lf'])-1)
        for pt_ind in range(begin_ind, end_ind+1):
            pt_x = float(out['lf'][pt_ind][line_ind][0][0])
            pt_y = float(out['lf'][pt_ind][line_ind][1][0])
            pt_x, boundary_x = correct_pt(pt_x, img_width)
            pt_y, boundary_y = correct_pt(pt_y, img_height)            
            if prev != [pt_x, pt_y]:
                polygon.append((pt_x, pt_y)) 
                prev = [pt_x, pt_y]
        for pt_ind in range(end_ind, begin_ind-1, -1):    
            pt_x = float(out['lf'][pt_ind][line_ind][0][1])
            pt_y = float(out['lf'][pt_ind][line_ind][1][1])
            pt_x, boundary_x = correct_pt(pt_x, img_width)
            pt_y, boundary_y = correct_pt(pt_y, img_height)
            if prev != [pt_x, pt_y]:
                polygon.append((pt_x, pt_y))
                prev = [pt_x, pt_y]
        
        polygon_list.append(polygon) 
        if len(polygon) < 3:
            print('WARNING: DEGENERATE POLYGON AT INDEX', len(polygon_list))
    return polygon_list

# Each polygon is a list of (x,y) tuples
def get_polygon_list_without_trim(out):
    img = cv2.imread(out["image_path"])
    img_height, img_width = img.shape[:2]
    polygon_list = []
    for line_ind in range(len(out['lf'][0])):
        polygon = []
        begin_ind = 0
        end_ind = len(out['lf'])-1
        prev = [-1, -1]
        
        for pt_ind in range(begin_ind, end_ind+1):
            pt_x = float(out['lf'][pt_ind][line_ind][0][0])
            pt_y = float(out['lf'][pt_ind][line_ind][1][0])
            pt_x, boundary_x = correct_pt(pt_x, img_width)
            pt_y, boundary_y = correct_pt(pt_y, img_height)
            if prev != [pt_x, pt_y]:
                polygon.append((pt_x, pt_y)) 
                prev = [pt_x, pt_y]
        for pt_ind in range(end_ind, begin_ind-1, -1):    
            pt_x = float(out['lf'][pt_ind][line_ind][0][1])
            pt_y = float(out['lf'][pt_ind][line_ind][1][1])
            pt_x, boundary_x = correct_pt(pt_x, img_width)
            pt_y, boundary_y = correct_pt(pt_y, img_height)
            if prev != [pt_x, pt_y]:
                polygon.append((pt_x, pt_y)) 
                prev = [pt_x, pt_y]

        if len(polygon) >= 3:
            polygon_list.append(polygon) 
    return polygon_list

# Each polygon passed as input is a list of (x,y) tuples
# Same for output
def percent_intersection(size, poly1, poly2): 
    im1 = Image.new(mode="1", size=size)
    draw1 = ImageDraw.Draw(im1)    
    draw1.polygon(poly1, fill=1)
    im2 = Image.new(mode="1", size=size)    
    draw2 = ImageDraw.Draw(im2)
    draw2.polygon(poly2, fill=1)   
    mask1 = np.asarray(im1, dtype=bool)
    mask2 = np.asarray(im2, dtype=bool)
    intersection_mask = mask1 & mask2
    #plt.imshow(intersection)
    intersection_area = intersection_mask.sum()    
    percent1 = intersection_area / mask1.sum()
    percent2 = intersection_area / mask2.sum()
    return intersection_area, percent1, percent2



def get_poly_no_overlap(img_name, poly_list, threshold=0.6):
    
    img=Image.open(img_name)
    size=img.size
    #polygons = [points.list_to_xy(p) for p in poly_list]
    polygons = poly_list
    del_list = []
    current = 0
    next_ind = current+1
    last_deleted = -1
    while next_ind<len(polygons):
        # Check these are not degernate polygons
        if len(polygons[current]) < 3:
            del_list.append(current)
            current, next_ind = (current+1, next_ind+1)
            continue
        if len(polygons[next_ind]) < 3:
            del_list.append(next_ind)
            next_ind += 1
            continue
        # End check
        overlap_area, percent1, percent2 = percent_intersection(size, 
                                                                polygons[current], 
                                                                polygons[next_ind])
        
        if percent1 > threshold or percent2 > threshold:
            to_del = current if percent1 > percent2 else next_ind
            current, next_ind = (current, next_ind+1) if percent1<percent2\
                                else (next_ind, next_ind+1)
            del_list.append(to_del)
            last_deleted = to_del
            #print('last deleted', to_del)
        else: # when no overlap is found
            current, next_ind = (current+1, next_ind+1)
            if current <= last_deleted:
                current = last_deleted + 1
                next_ind = current + 1
    all_ind = set(range(len(poly_list)))
    good_ind = all_ind.difference(set(del_list))
    poly_non_overlapping = [poly_list[i] for i in good_ind]
    return del_list, poly_non_overlapping



def dump_polygons_json(out, polygons = None, filename=None):
    if filename is None:
        filename = out["image_path"][:-3] + "json"
    if polygons is None:
        polygons = get_polygon_list(out)
    lf_dict = {}
    for ind, poly in enumerate(polygons):
        lf_dict['line_' + str(ind+1)] = points.xy_to_list(poly)

    with open(filename, 'w') as fout:
        json_dumps_str = json.dumps(lf_dict, indent=2)
        #print('....json_dumps_str', json_dumps_str)
        print(json_dumps_str, file=fout)   
        
def write_json_file(out, poly_list = None, json_file=None):
    if poly_list is None:
        poly_list = get_polygon_list(out)
    dump_polygons_json(out, poly_list, json_file)
    
def write_text_file(out, predicted_text, filename=None):
    if filename is None:
        filename = out["image_path"][:-3] + "txt"
    prediction_para = '\n'.join(predicted_text)
    with open(filename, 'w') as f:
        f.write(prediction_para)
        
# won't flip the polygons...only the image        
def draw_image_with_poly(directory, image, poly, convert=True, flip=False):
    img = cv2.imread(os.path.join(directory, image))
    if flip:
        img = cv2.flip(img, 1)
    plt.imshow(img)
    colors = ['red', 'green', 'blue']
    
    for ind, p in enumerate(poly):
        if convert:
            p = points.list_to_xy(p)
        points.draw_poly(plt, p, colors[ind%3])
        plt.text(p[-1][0], p[-1][1], str(ind))

        
arabic_reshaper_configuration = {
    'delete_harakat': True,
    'support_ligatures': True
}
        

def remove_diacritics(txt):
    reshaper = ArabicReshaper(configuration=arabic_reshaper_configuration)
    txt_without_diacritics = reshaper.reshape(txt)
    return(txt_without_diacritics)

def get_cer_matrix(gt_list, prediction_list):
    total_gt = len(gt_list)
    total_predictions = len(prediction_list)
    cer_matrix = np.zeros((total_gt, total_predictions))
    
    for ind_g, g in enumerate(gt_list):
        for ind_p, p in enumerate(prediction_list):
            cer_matrix[ind_g, ind_p] = error_rates.cer(g, p)
            if cer_matrix[ind_g, ind_p] > 1:
                cer_matrix[ind_g, ind_p] = 1
    
    return cer_matrix



# https://en.wikipedia.org/wiki/Dynamic_time_warping
def get_dtw(dist_matrix):
    r,c = dist_matrix.shape
    #dtw = np.zeros((r+1, c+1))
    #for i in range(dtw.shape[0]):
        #for j in range(dtw.shape[1]):
    #        dtw[i, j] = np.inf
    
    dtw = np.full((r+1, c+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, dtw.shape[0]):
        for j in range(1, dtw.shape[1]):
            dtw[i, j] = dist_matrix[i-1, j-1] + np.min([dtw[i-1, j], dtw[i, j-1],
                                                    dtw[i-1, j-1]])
    return dtw[-1, -1]
    
    
    


def get_cers(gt_list, prediction_list, no_diacritics=False):
    
    if no_diacritics:
        gt_list = [remove_diacritics(g) for g in gt_list]
        prediction_list = [remove_diacritics(p) for p in prediction_list]
    
    cer_matrix = get_cer_matrix(gt_list, prediction_list)
    cer_gt = cer_matrix.min(axis=1).flatten()
    cer_p = cer_matrix.min(axis=0).flatten()
    cer_dtw = get_dtw(cer_matrix)
    
    gt = '\n'.join(gt_list)
    pred = '\n'.join(prediction_list)
    cer_para = error_rates.cer(gt, pred)
    
    return cer_dtw, np.sum(cer_gt), np.sum(cer_p), cer_para

def get_cers_wer(gt_list, prediction_list, no_diacritics=False):
    
    if no_diacritics:
        gt_list = [remove_diacritics(g) for g in gt_list]
        prediction_list = [remove_diacritics(p) for p in prediction_list]
    
    cer_matrix = get_cer_matrix(gt_list, prediction_list)
    cer_gt = cer_matrix.min(axis=1).flatten()
    cer_p = cer_matrix.min(axis=0).flatten()
    cer_dtw = get_dtw(cer_matrix)
    
    gt = '\n'.join(gt_list)
    pred = '\n'.join(prediction_list)
    cer_para = error_rates.cer(gt, pred)
    wer_para = error_rates.wer(gt, pred)
    
    return cer_dtw, np.sum(cer_gt), np.sum(cer_p), cer_para, wer_para
        
def flip_polygon(img_file, poly_list):
    img = cv2.imread(img_file)
    h, w = img.shape[:2]
    flipped_poly_list = []
    for p in poly_list:
        flipped = [(w-x, y) for (x, y) in p]
        flipped_poly_list.append(flipped)
    return flipped_poly_list

# This will create annotations that can be used in scribeArabic annotation tool
def create_annotations_json(predicted_text, poly_list):
    if len(predicted_text) != len(poly_list):
        print("POLYGONG LIST LEN Not same as PREDICTED TEXT LEN")
        slkfj
    page_json = dict()
    for ind, (ocr, poly) in enumerate(zip(predicted_text, poly_list)):
        pts = points.xy_to_list(poly)
        line_key = f'line_{ind+1}'
        page_json[line_key] = dict()
        page_json[line_key]['coord'] = pts
        page_json[line_key]['text'] = ocr
    return page_json
