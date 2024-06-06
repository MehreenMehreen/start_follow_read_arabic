import sys
sys.path.append('../py3/')
import os
import json
import yaml
import pandas as pd
import numpy as np
import torch
import hw
from hw import cnn_lstm
from utils import string_utils, error_rates
import cv2

HT = 60

def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v
    return idx_to_char
        
    

def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)      
    return config



def load_HW(hw_network_config, pt_filename):
    HW = cnn_lstm.create_model(hw_network_config)
    hw_state = torch.load(pt_filename)
    HW.load_state_dict(hw_state)
    
    device = torch.device("cuda")
    HW.to(device)
    return HW
    
def get_predicted_str(HW, img_file, idx_to_char, device="cuda", flip=False, 
                      show=False, read_image=True, img=None, tokenizer=None):    


    device = torch.device(device)
    if read_image:
        img = cv2.imread(img_file)
    ht, width = img.shape[:2]
    
    if ht != HT:
        new_width = int(width/ht*HT)
        img = cv2.resize(img, (new_width, HT))
    if show:
        
        plt.imshow(img)
        plt.show()
    if flip:
        img = np.flip(img, axis=1)
    img = img.astype(np.float32)
    img = img / 128.0 - 1.0
    img = np.expand_dims(img, 0)
    img = img.transpose([0,3,1,2])
    img = torch.from_numpy(img)


    IMG = img.to(device)
    #print('img size is', img.size())
    
    preds = HW(IMG).cpu()

    out = preds.permute(1,0,2)
    out = out.data.numpy()
    logits = out[0,...]
    pred, raw_pred = string_utils.naive_decode(logits)
    if tokenizer is None:
        pred_str = string_utils.label2str_single(pred, idx_to_char, False)
    else:
        pred_str = tokenizer.decode(pred)
    return pred_str


def write_csv_all_predictions(config_file, suffix="", device="cuda", flip=False, pt_file='hw.pt', 
                              test_file_to_use="", result_file="", tokenizer=None):   
    
    result_df = pd.DataFrame(columns=["image", "ground_truth", "prediction", "CER", "WER"])
    config = get_config(config_file)    
    idx_to_char = load_char_set(config['network']['hw']['char_set_path'])
    if 'hw_to_save' in config['pretraining'].keys():
        pt_file = config['pretraining']['hw_to_save']
    else:
        pt_file = pt_file
    pt_filename = os.path.join(config['pretraining']['snapshot_path'], pt_file)
    
    config["network"]["hw"]["num_of_outputs"] = len(idx_to_char) + 1
    if tokenizer is not None:
        config["network"]["hw"]["num_of_outputs"] = tokenizer.get_vocab_size()
        pt_filename = os.path.join(config['pretraining']['snapshot_path'], f"hw_tokenizer_{tokenizer.get_vocab_size()}.pt")
    
    if len(suffix) > 0:
        pt_filename = pt_filename[:-3] + suffix + '.pt'
    print('...Using snapshot', pt_filename)    
    HW = load_HW(config['network']['hw'], pt_filename)
    device = torch.device(device)
    HW.to(device)
    HW.eval()
    
    if test_file_to_use == "":
        test_json_file = config['testing']['test_file']
    else:
        test_json_file = test_file_to_use
        print('Using test file', test_json_file)
    
    with open(test_json_file) as f:
        json_obj = json.load(f)
        for ind, obj in enumerate(json_obj):
            #obj is a list of: [jsonfile imgfile]
            #open the json file and get a list of predictions
            with open(obj[0]) as f:
                image_list = json.load(f)

            for record in image_list:
                # If not gt in file
                if record['gt'] == 'None' or isinstance(record['gt'], float) or record['gt'] == 'nan' or len(record['gt']) == 0:
                    print(type(record['gt']), record['gt'], record['hw_path'])
                    continue
                print(record['hw_path'])
                predicted_str = get_predicted_str(HW, record['hw_path'], idx_to_char, device=device, 
                                                  flip=flip, tokenizer=tokenizer)
                
                cer = error_rates.cer(record['gt'], predicted_str)
                wer = error_rates.wer(record['gt'], predicted_str)
                result_df.loc[len(result_df)] = [record['hw_path'], record['gt'], 
                                                predicted_str, cer, wer]       

    if len(result_file) == 0:
        result_file = config_file.replace("config", "result")
        result_file = result_file.replace("yaml", "csv")
    if len(suffix)>0:
        result_file = result_file[:-4] + suffix + '.csv'
    result_df.to_csv(result_file, index=False)
    return result_df
    
   