import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw


# img_file is the filename with full path
# points is a string of coordinate points read from XML
def generate_cropped_region(img_file, points):
    # Process points
    points_list = points.split(' ')
    xy_pts = [(int(points.split(",")[0]),
               int(points.split(",")[1])) for points in points_list]
    pts = np.array(xy_pts)
    img_obj = Image.open(img_file) 
    img = np.array(img_obj)
    # Crop the image
    [min_x, min_y] = np.min(pts, axis=0)
    [max_x, max_y] = np.max(pts, axis=0)
    cropped_img = img[min_y:max_y+1, min_x:max_x+1, :]
    cropped_img_obj = Image.fromarray(cropped_img)
    return cropped_img_obj, (min_x, min_y), (max_x, max_y)

# pts is numpy 2D points array    
# img also numpy arrray/cv2 array
def generate_cropped_image(img, pts):    
    img_obj = Image.fromarray(img)
    
    # Create a polygonal mask
    mask = Image.new('L', (img_obj.width, img_obj.height), color=0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.polygon(list(pts.flatten()), fill=255)
    mask = np.array(mask).astype(bool)
    # Choose the polygonal area from image
    output_img = np.zeros_like(img)
    output_img[mask] = img[mask]
    # Crop the image
    [min_x, min_y] = np.min(pts, axis=0).astype(int)
    [max_x, max_y] = np.max(pts, axis=0).astype(int)
    cropped_img = output_img[min_y:max_y+1, min_x:max_x+1, :]

    return cropped_img


def draw_poly(plt, xy_pts, color='green'):
    
    plt.gca().add_patch(Rectangle(xy_pts[0], 10, 10, facecolor='yellow'))
    for pts1, pts2 in zip(xy_pts, xy_pts[1:]):
        #img = color_rect(img,pts1[0], pts1[1], pts2[0], pts2[1])
        draw_line(plt, pts1[0], pts1[1], pts2[0], pts2[1], color)
        draw_line(plt, xy_pts[0][0], xy_pts[0][1], 
                   xy_pts[-1][0], xy_pts[-1][1], color)
        
def draw_baseline(plt, xy_pts, color='red'):

    plt.gca().add_patch(Rectangle(xy_pts[0], 100, 100, facecolor='blue'))
    for pts1, pts2 in zip(xy_pts, xy_pts[1:]):
        #img = color_rect(img,pts1[0], pts1[1], pts2[0], pts2[1])
        draw_line(plt, pts1[0], pts1[1], pts2[0], pts2[1], color=color)
        
def draw_line(plt_obj, x1, y1, x2, y2, color='g'):
    plt_obj.plot([x1, x2], [y1, y2], color=color, linewidth=1)

# The argument points is a string. 
# Function returns a list of (x,y) tuples
def get_xy_pts(points):
    points_list = points.split(' ')
    xy_pts = [(int(points.split(",")[0]),
               int(points.split(",")[1])) for points in points_list]
    return xy_pts

# bbox is not necessarily a polygon or rectangle. Just a list of (x,y) tuples
# If apply_correction is True then all points are restricted to lie within 
# top left and bottom right
def add_offset_to_polygon(bbox, offset, apply_correction=False, 
                          top_left=[], bottom_right=[]):
    new_bbox = []
    for i,coord in enumerate(bbox):
        new_bbox.append((coord[0]+offset[0], coord[1]+offset[1]))
    if apply_correction:
        top_left = np.array(top_left)
        bottom_right = np.array(bottom_right)
        pts = np.array(new_bbox)
        for j in [0, 1]:
            ind = np.where(pts[:, j] > bottom_right[j])
            pts[ind, j] = bottom_right[j]
        for j in [0, 1]:
            ind = np.where(pts[:, j] < top_left[j])
            pts[ind, j] = top_left[j]
        new_bbox = list(map(tuple, pts))
            
    return new_bbox

def add_offset_to_polygon_list(polygon_list, offset):
    new_polygon_list = []
    for poly in polygon_list:
        new_poly = add_offset_to_polygon(poly, offset)
        new_polygon_list.append(new_poly)
    return new_polygon_list
        
def combine_poly(poly1, poly2):
    main_poly = [poly1[0], poly1[1]]
    if poly1[1][0] != poly2[0][0] or poly1[1][1] != poly2[0][1]:
        main_poly.append(poly2[0])
    main_poly.extend(poly2[1:3])
    if poly1[2][0] != poly2[3][0] or poly1[2][1] != poly2[3][1]:
        main_poly.append(poly2[3])
    main_poly.extend(poly1[2:])    
    return main_poly
    
# Get left, upper, right, lower min_x, min_y, max_x, max_y
def get_max_min_polygon(polygon):
    x_list = [x[0] for x in polygon]
    y_list = [y[1] for y in polygon]
    return min(x_list), min(y_list), max(x_list), max(y_list)    

def add_offset_to_baseline(baseline, offset):
    new_baseline = []
    for pts in baseline:
        new_baseline.append((pts[0]+offset[0], pts[1]+offset[1]))
    return new_baseline    


def add_offset_to_baseline_list(baseline_list, offset):
    new_list = []
    for b in baseline_list:
        new_b = add_offset_to_baseline(b, offset)
        new_list.append(new_b)
    return new_list
        
def combine_baseline(base1, base2):
    #combined = [base1[0], base1[1], base2[0], base2[1]]
    combined = [base2[1], base2[0], base1[1], base1[0]]
    return combined


def get_x_y(polygon):
    x_list = [x[0] for x in polygon]
    y_list = [y[1] for y in polygon]
    return x_list, y_list


# num_pts is number of points on baseline to get
def get_baseline_regression(poly_pts, num_pts=10, deg=1):
    if len(poly_pts) <= 4:
        deg = 1
    x, y = get_x_y(poly_pts)
    model = (np.polyfit(x, y, deg))
    p = np.poly1d(model)
    # get the x against which we want y
    x1, y1, x2, y2 = get_max_min_polygon(poly_pts)
    num_pts = min(num_pts, x2-x1+1)
    num_pts = int(num_pts)
    x = np.linspace(x1, x2, num_pts, endpoint=True, dtype=int)
    y = p(x)
    baseline = [(a, b) for a,b in zip(x, y)]
    return baseline



# Given a coordinates list return xy tuples
def list_to_xy(coord_list):
    xy_list = []
    for ind in range(0, len(coord_list), 2):
        xy_list.append((coord_list[ind], coord_list[ind+1]))
    return xy_list

# Given an (x,y) list of tuples, return a flat list
def xy_to_list(tuples_list):
    flat_list = [x for pair in tuples_list for x in pair]
    return flat_list


# x is a list of points
# y is a cooresponding list of points
def get_baseline_from_xy(x, y, num_pts=10, deg=1):
    if len(x) <= 4:
        deg = 1
    
    model = (np.polyfit(x, y, deg))
    p = np.poly1d(model)
    # get the x against which we want y
    x1, y1, x2, y2 = get_max_min_polygon(poly_pts)
    num_pts = min(num_pts, x2-x1+1)
    x = np.linspace(x1, x2, num_pts, endpoint=True, dtype=int)
    y = p(x)
    baseline = [(a, b) for a,b in zip(x, y)]
    return baseline

# Here img is the 2D numpy array 
# xy_pts is a list of (x,y) tuples
def generate_cropped_region_from_polypts(img, xy_pts):
    pts = np.array(xy_pts)
    
    # Crop the image
    [min_x, min_y] = np.ceil(np.min(pts, axis=0)).astype(int)
    [max_x, max_y] = np.floor(np.max(pts, axis=0)).astype(int)
    cropped_img = img[min_y:max_y+1, min_x:max_x+1, :]
    cropped_img_obj = Image.fromarray(cropped_img)
    return cropped_img_obj, (min_x, min_y), (max_x, max_y)

# Will generate a line image by using xy_pts as a mask
def generate_line_image(img, xy_pts):

    pts = np.array(xy_pts)
    [min_x, min_y] = np.ceil(np.min(pts, axis=0)).astype(int)
    [max_x, max_y] = np.floor(np.max(pts, axis=0)).astype(int)
    (width, ht) = (max_x-min_x+1, max_y-min_y+1)
    xy_pts = add_offset_to_polygon(xy_pts, (-min_x, -min_y))
    img = img[min_y:max_y+1, min_x:max_x+1, :]
    #print('min_y:max_y+1, min_x:max_x+1, :', min_y, max_y+1, min_x, max_x+1)
    img_obj = Image.fromarray(img)
    
    draw_img = ImageDraw.Draw(img_obj)
    # Create a polygonal mask
    mask = Image.fromarray(np.zeros((img.shape[0], img.shape[1])))
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.polygon(xy_pts, fill='white')
    mask = np.array(mask).astype(bool)
    # Choose the polygonal area from image
    
    output_img = np.zeros_like(img)+255
    output_img[mask] = img[mask]
    output_img = Image.fromarray(output_img)
    return output_img, xy_pts


# restrict coordinates to lie between 0 and max (included)
def restrict_pts(pts, max_p):
    pts = [(max(0, x), max(0, y)) for (x,y) in pts]
    pts = [(min(max_p[0], x), min(max_p[1], y)) for (x,y) in pts]
    return pts

# assuming poly_pts is a list of (x,y) tuples
# This will add more points by interpolating between two points
def expand_poly(poly_pts, min_x_increment=10):
    poly_pts = np.array(poly_pts).astype(int)
    new_poly = []
    #    for ind, (curr, nxt) in enumerate(zip(poly_pts[:-1], poly_pts[1:])):

    for ind, curr in enumerate(poly_pts):
        nxt = poly_pts[(ind+1)%len(poly_pts)]
        if np.abs(nxt[0] - curr[0]) < min_x_increment:
            new_poly.append(curr)
            new_poly.append(nxt)
            continue
        x1, x2 = curr[0], nxt[0]
        y1, y2 = curr[1], nxt[1]
        new_poly.append(curr)
        increment = -1*min_x_increment if nxt[0] < curr[0] else min_x_increment
        for x in range(curr[0]+1, nxt[0], increment):
            slope = float(y2-y1)/(x2-x1)
            y = slope*(x - x2) + y2
            new_poly.append((x, y))
        new_poly.append(nxt)
    return new_poly
            

# Chunks_len ignored if chunk_len_auto is True
def get_baseline_chunks(poly_pts, chunks_len=300, chunk_len_auto=True):
    baseline = []
    poly_pts = expand_poly(poly_pts)
    
    
    p = np.array(poly_pts)
    
    max_x, max_y = np.max(p, 0)
    min_x, min_y = np.min(p, 0)
    
    # Decide chunks_len
    if chunk_len_auto:
        if (len(poly_pts) >= 250):
            total_chunks = 5
        else:
            total_chunks = int(np.ceil(len(poly_pts)/50))
        chunks_len = int((max_x-min_x)/total_chunks)
    else:
        total_chunks = int((max_x-min_x)/chunks_len)
    
    #print('expanded len', len(poly_pts), 'total chunks', total_chunks)
    for i in range(1, total_chunks+1):
        p1 = [pt for pt in p if (pt[0]-min_x)>=(i-1)*chunks_len and (pt[0]-min_x)<i*chunks_len]
        #print(p1)
        if i == total_chunks:
            p1 = [pt for pt in p if (pt[0] - min_x)>=(i-1)*chunks_len]
        b = get_baseline_regression(p1, num_pts=12)
        
        # Points are in ascending order (increasing x - left to right) 
        if len(baseline) != 0:
            # This will smooth out the line
            # Get rid of last 4 points and connect the point with next 4 points
            baseline = baseline[:-4]
            baseline.extend(b[4:])
            if i == total_chunks:
                baseline.extend(b[-1:])
        else:
            baseline = b

    # Make sure baseline does not repeat

    prev_pt = baseline[0]
    baseline_clean = [prev_pt]
    for b in baseline[1:]:
        if b != prev_pt:
            baseline_clean.append(b)
            prev_pt = b
    return baseline_clean
    

# Making sure a value is not outside a boundary or has negative value
def correct_pt(value, max_value):
    if value < 0:
        return 0
    if value > max_value:
        return max_value
    return value

# Assume poly is list of (x, y) tuples or [x, y] list
# Will retrieve a vertically oriented baseline
# Will return top to bottom and bottom to top if reversed is true
def get_vertical_baseline(poly, reversed=False):
    flipped_poly = [(y, x) for (x, y) in poly] 
    baseline = get_baseline_chunks(flipped_poly)
    # Flip back
    baseline = [(y, x) for (x, y) in baseline]
    if reversed:
        baseline.sort(key=lambda x: x[1], reverse=True)
    return baseline

# Check the polygon is valid
# If x coord or y coord don't change, its not valid
def valid_poly(poly_pts):
    if len(poly_pts) <= 2:
        return False
    x = [pt[0] for pt in poly_pts]
    y = [pt[1] for pt in poly_pts]
    
    if np.max(x) - np.min(x) <= 1e-2:
        return False
    if np.max(y) - np.min(y) <= 1e-2:
        return False
    return True