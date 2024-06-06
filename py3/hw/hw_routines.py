# Modified from https://github.com/cwig/start_follow_read/  filename: hw_pretraining.py
# Implements conditional instance normalization for different writers
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torch.nn import CTCLoss
from torch import nn

import hw
from hw import hw_dataset
from hw import cnn_lstm_cin, cnn_lstm
from hw.hw_dataset import HwDataset
import pickle

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml

from utils.dataset_parse import load_file_list
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import pandas as pd
import plot_log

# Keep as global to avoid memory leaks
HW = None
IMG = None

def load_hw_network(config):
    global HW
    transfer_learn = config['cin']['transfer_learn'] 
    print('transfer_learn in config', transfer_learn)
    print('transfer learn', transfer_learn)
    if transfer_learn:
        HW = cnn_lstm_cin.load_CIN_network_from_CRNN(config)
        print('HW network loaded from CRNN')
        return
    
    HW = cnn_lstm_cin.create_model(config)
    # Check in config file. If path is there then load from there
    if len(config['pretraining']['pretrained_network']) > 0:
        hw_path = config['pretraining']['pretrained_network']
        hw_state = safe_load.torch_state(hw_path)
        HW.load_state_dict(hw_state)
        print("HW loaded from state dict")
    return
    

    
def load_charset(hw_network_config):
    char_set_path = hw_network_config['char_set_path']

    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v
    return char_set, idx_to_char


def setup_dataloaders(char_set, config):
    pretrain_config = config['pretraining']
    network_config = config['network']['hw']
    training_set_list = load_file_list(pretrain_config['training_set'])
    writer_list = config['cin']['writer_list']
    
    train_dataset = HwDataset(training_set_list,
                              char_set['char_to_idx'], augmentation=True,
                              img_height=network_config['input_height'], 
                              flip_image=True, absolute_path=True, writer_list=writer_list)

    train_dataloader = DataLoader(train_dataset,
                                 batch_size=pretrain_config['hw']['batch_size'],
                                 shuffle=True, num_workers=0, drop_last=True,
                                 collate_fn=hw_dataset.collate)

    batches_per_epoch = int(pretrain_config['hw']['images_per_epoch']/pretrain_config['hw']['batch_size'])
    train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

    valid_set_list = load_file_list(pretrain_config['validation_set'])
    valid_dataset = HwDataset(valid_set_list,
                             char_set['char_to_idx'],
                             img_height=network_config['input_height'], 
                             flip_image = True, absolute_path=True, writer_list=writer_list)

    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size=pretrain_config['hw']['batch_size'],
                                 shuffle=False, num_workers=0,
                                 collate_fn=hw_dataset.collate)
    return train_dataloader, valid_dataloader

def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return config
    
def get_cer_loss(dataloader, idx_to_char, DEVICE):
    HW.eval()
    dtype = torch.cuda.FloatTensor
    sum_cer_loss = 0.0
    steps = 0.0
    for x in dataloader:
        line_imgs = x['line_imgs'].type(dtype).to(DEVICE)
        labels =  x['labels']
        label_lengths = x['label_lengths']
        writers = x['writers']

        preds = HW(line_imgs, writers).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_cer_loss += cer
            steps += 1
    return sum_cer_loss/steps

def save_training_history(train_loss, ctc_loss, valid_loss, log_file, log_file_prefix=""):
    path, filename = os.path.split(log_file)
    filename = log_file_prefix + filename
    log_file = os.path.join(path, filename)
    
    train_stat = {'train_loss': train_loss, 'ctc_loss': ctc_loss, 'valid_loss': valid_loss}
    with open(log_file, "w") as outfile:
        strm = csv.writer(outfile)
        for key, val in train_stat.items():
            strm.writerow([key, val])

def save_network(config, prefix=""):
    if not os.path.exists(config['pretraining']['snapshot_path']):
        os.makedirs(config['pretraining']['snapshot_path'])
    torch.save(HW.state_dict(), 
               os.path.join(config['pretraining']['snapshot_path'], 
                            prefix+config['pretraining']['hw_to_save']))

    
    
def train(config, train_dataloader, valid_dataloader, idx_to_char, DEVICE, total_epochs):
# To be able to access globally
    global train_loss, valid_loss, ctc_loss
    train_loss = []
    valid_loss = []
    ctc_loss = []  
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(HW.parameters(), lr=config['pretraining']['hw']['learning_rate'])
    dtype = torch.cuda.FloatTensor

    lowest_loss = np.inf
    cnt_since_last_improvement = 0
    for epoch in range(total_epochs):
        first = True
        
        train_cer = 0.0
        train_steps = 0.0
        HW.to(DEVICE)
        HW.train()
        total_ctc_loss = 0.0
        count = 0
        for i, x in enumerate(train_dataloader):
            #if i==2:
            #    break
            #print("\n layer 0 wts: ", HW.rnn.multiple_embeddings[0].weight, 'bais', HW.rnn.multiple_embeddings[0].bias)
            #print("\n layer 1 wts: ", HW.rnn.multiple_embeddings[1].weight, 'bias', HW.rnn.multiple_embeddings[1].bias)    
            line_imgs = x['line_imgs'].type(dtype).to(DEVICE)
            labels =  x['labels']
            label_lengths = x['label_lengths']
            writers = x['writers']
            #print('...writers', writers)
            preds = HW(line_imgs, writers).cpu()
            output_batch = preds.permute(1,0,2)
            out = output_batch.data.cpu().numpy()

            for i, gt_line in enumerate(x['gt']):
                logits = out[i,...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                train_cer += cer
                train_steps += 1


            batch_size = preds.size(1)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            loss = criterion(preds, labels, preds_size, label_lengths)
            total_ctc_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = count + 1
        # After one training iteration (not epoch)    
        valid_cer = get_cer_loss(valid_dataloader, idx_to_char, DEVICE)
        train_loss.append(train_cer/train_steps)
        ctc_loss.append(total_ctc_loss.item())
        valid_loss.append(valid_cer)        
        cnt_since_last_improvement += 1

        if lowest_loss > valid_cer:
            cnt_since_last_improvement = 0
            lowest_loss = valid_cer
            print ("Saving Best with valid_cer", valid_cer)
            save_network(config)
            save_training_history(train_loss, ctc_loss, valid_loss, config['pretraining']['hw_log_file'])
        if cnt_since_last_improvement >= config['pretraining']['hw']['stop_after_no_improvement'] and lowest_loss<0.35:
            break
        if epoch%5 == 0:
            print('Epoch: {}, Train error: {:.2f}, Valid error: {:.2f}'.format(epoch, train_cer/train_steps, valid_cer))
    # Save the last state        
    save_network(config, "last_")    
    save_training_history(train_loss, ctc_loss, valid_loss, config['pretraining']['hw_log_file'], "last_")

    print('Done training...')
    
def get_predicted_str(img_file, idx_to_char, writer, DEVICE, show=False):    
    global IMG
    HT = 60
    
    img = cv2.imread(img_file)
    ht, width = img.shape[:2]
    
    if ht != HT:
        new_width = int(width/ht*HT)
        img = cv2.resize(img, (new_width, HT))
    if show:
        
        plt.imshow(img)
        plt.show()
    img = np.flip(img, axis=1)
    img = img.astype(np.float32)
    img = img / 128.0 - 1.0
    img = np.expand_dims(img, 0)
    img = img.transpose([0,3,1,2])
    img = torch.from_numpy(img)

    IMG = img
    IMG = img.to(DEVICE)
    #print('img size is', img.size())
    
    preds = HW(IMG, writer).cpu()

    out = preds.permute(1,0,2)
    out = out.data.numpy()
    logits = out[0,...]
    pred, raw_pred = string_utils.naive_decode(logits)
    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
    return pred_str

def get_all_predictions(CONFIG_LIST, DEVICE):   
    global HW
    
    for config_file in CONFIG_LIST:
        result_df = pd.DataFrame(columns=["image", "writer", "ground_truth", "prediction", "CER"])
        HW = None
        config = get_config(config_file)    
        char_set, idx_to_char = load_charset(config['network']['hw'])
        config['cin']['transfer_learn'] = 0
        config['pretraining']['pretrained_network'] = os.path.join(config['pretraining']['snapshot_path'], 
                                                                   config['pretraining']['hw_to_save'])
        writer_list = config['cin']['writer_list']
        load_hw_network(config)
        HW.to(DEVICE)
        json_file = config['testing']['test_file']
        with open(json_file) as f:
            json_obj = json.load(f)
            for ind, obj in enumerate(json_obj):
                # obj is a list of: [jsonfile not_used]
                # open the json file and get a list of predictions
                with open(obj[0]) as f:
                    image_list = json.load(f)
                
                for record in image_list:
                    
                    writer = [writer_list.index(record['writer'])]
                    predicted_str = get_predicted_str(record['hw_path'], idx_to_char, writer, DEVICE)
                    cer = error_rates.cer(record['gt'], predicted_str)
                    result_df.loc[len(result_df)] = [record['hw_path'], record['writer'], record['gt'], 
                                                    predicted_str, cer]       
                    
        result_file = config_file.replace("config", "result")
        result_file = result_file.replace("yaml", "csv")
        result_df.to_csv(result_file)
        
        
def get_test_cer(json_file, idx_to_char, writer_list):
    global HW
    cer_list = []
    sum_cer = 0
    cer_dict = dict()
    HW.eval()
    
#    config['cin']['writer_list'] = ["shebil", "Rihani"]
    writer = [0] if writer_list[0] in json_file else [1]
#    print('writer', writer)
    with open(json_file) as f:
        json_obj = json.load(f)
        for ind, obj in enumerate(json_obj):
            predicted_str = get_predicted_str(obj['hw_path'], idx_to_char, writer)
    #        print(obj['gt'])
            cer = error_rates.cer(obj['gt'], predicted_str)
            cer_list.append(cer)
            sum_cer += cer
            cer_dict[obj['hw_path']] = cer
    avg_cer = sum_cer/len(json_obj)
    return avg_cer, cer_list, cer_dict, writer

def get_error_multiple_writers():
    global HW
    # Test for single file
    config_file = '/home/msaeed3/mehreen/datasets/MoiseK/batch_01_writers/multiple_all/cin_layer-2-4/config_{}_cin.yaml'.format(2)    
    config = get_config(config_file)    

    writer_list = config['cin']['writer_list']
    avg_cer_list = np.empty((0, len(writer_list)))
    for w in range(2, 11):
        print('processing config', w)
        config_file = '/home/msaeed3/mehreen/datasets/MoiseK/batch_01_writers/multiple_all/cin_layer-2-4/config_{}_cin.yaml'.format(w)    
        config = get_config(config_file)    

        char_set, idx_to_char = load_charset(config['network']['hw'])
        HW = None
        config['cin']['transfer_learn'] = 0
        config['pretraining']['pretrained_network'] = os.path.join(config['pretraining']['snapshot_path'], 
                                                                   config['pretraining']['hw_to_save'])
        load_hw_network(config, HW)
        json_file = config['testing']['test_file']
        with open(json_file) as f:
            json_obj = json.load(f)
            cer_one_config = np.zeros((1, len(writer_list)))
            for ind, obj in enumerate(json_obj):
                avg_cer, cer_list, cer_dict, writer = get_test_cer(obj[0], idx_to_char, writer_list)   
                cer_one_config[0, writer] = avg_cer
                print('avg_cer', avg_cer)
        avg_cer_list = np.vstack((avg_cer_list, cer_one_config))
        print('avg_cer_list', avg_cer_list)
    return avg_cer_list
        
