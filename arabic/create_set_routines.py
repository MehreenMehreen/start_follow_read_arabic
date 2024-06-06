import json
import os

# Gets all the available json objects
def get_all_data(directory, batch_list):
    all_json = []
    
    for batch in batch_list:
        json_file = os.path.join(directory, batch + '.json')
        with open(json_file, 'r') as fin:
            json_obj = json.load(fin)
        all_json.extend(json_obj)
        
    return all_json

# Get the name of base directory when given basefilename
def get_basename(img_name):
    head, tail = os.path.split(img_name)
    basename = 'base_' + tail
    basename = basename[:-4]
    return basename

# Given an image file, search for the corresponding json file in the list of pretrain_dirs
def get_pretrain_json_filename(img_name, pretrain_dirs):
    #directory, img_name = os.path.split(img_name)
    base_dir = get_basename(img_name) + '/'  
    json_name = base_dir[:-1] + '.json'
    #print('base_dir:', base_dir)
    for d in pretrain_dirs:
        if os.path.exists(os.path.join(d, base_dir)):
            json_path = os.path.join(d, base_dir, json_name)
            if not os.path.exists(json_path):
                print('Json not found')
                return None
            return json_path
    return None

def get_pretrain_img_filename(input_json_path, all_pretrain_json):
    
    output_img_path = None
    
    for [json_file, img_file] in all_pretrain_json:
         if input_json_path == json_file:
                output_img_path = img_file
               
    return output_img_path

# Return only file names in a list. No path
# Also return a dictionary that holds the json and img name for each image name
# If json_list is None then json_file will be read. Otherwise input will be the json list
def get_img_list(json_file, json_list=None):
    file_dict = dict()
    file_list = []
    if json_list is None:
        with open(json_file) as fin:
            json_list = json.load(fin)
    # Get only filenames. No path
    for [json_file, img_file] in json_list:
        file = os.path.split(img_file)[1]
        file_dict[file] = {'json':json_file, 'img':img_file}
        file_list.append(file)
    return file_list, file_dict