# Code is modified from https://github.com/cwig/start_follow_read
# Train HW by loading from pre-trained and changing batchnorm to instance norm
import sys
sys.path.append('py3/')
sys.path.append('py3/hw/')
sys.path.append('arabic')
import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import hw
from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset
import pickle
from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load
import numpy as np
import cv2
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
import log_routines
import copy

def get_test_loss():
    sum_loss = 0.0
    steps = 0.0
    hw.eval()

    for x in test_dataloader:
        line_imgs = x['line_imgs'].to(DEVICE)
        labels =  x['labels']
        label_lengths = x['label_lengths']

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1
    return sum_loss/steps

def create_hw_from_pretrained(path, config, total_outputs):
    # Get the hw network from state dictionary
    hw_state = torch.load(path)
    pretrained_out_size = hw_state['rnn.embedding.bias'].shape[0]
    pretrained_config = copy.deepcopy(config)
    
    pretrained_config['num_of_outputs'] = pretrained_out_size
    pretrained_config['use_instance_norm'] = False
    print('pretrained_config', pretrained_config)
    pretrained_hw = cnn_lstm.create_model(pretrained_config)
    pretrained_hw.load_state_dict(hw_state)
    # Change number of outputs
    pretrained_hw.rnn.embedding = torch.nn.Linear(pretrained_hw.rnn.rnn.hidden_size*pretrained_hw.rnn.rnn.num_layers, 
                                            total_outputs)
    # Change batch norm layers
    print('config', config)
    # Change batch norm layers
    for name, module in pretrained_hw.cnn.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(pretrained_hw.cnn, name, torch.nn.InstanceNorm2d(module.num_features))    
    return pretrained_hw


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
       
    old_stdout = sys.stdout
    dump_file = log_routines.get_std_dump_filename(config_file, "hw")
    #sys.stdout = open(dump_file, 'w')
        
    old_stderr = sys.stderr
    dump_file = log_routines.get_std_dump_filename(config_file, "hw_err")
    #sys.stderr = open(dump_file, 'w')
           
        
init_training_time = time.time()
allowed_time = config['training']['hw']['reset_interval']     
    

train_loss = []
valid_loss = []
ctc_loss = []

if os.path.exists(config['pretraining']['snapshot_path']+'/hw.pt'):
    TRANSFER_HW = False
    LOAD_HW = True
else:
    TRANSFER_HW = True
    LOAD_HW = False

LOG_FILE = config['pretraining']['hw']['log_file']

train_stat = log_routines.read_log_dict(LOG_FILE)
if train_stat is not None:
    #train_loss.extend(train_stat['train_loss'])
    valid_loss.extend(train_stat['valid_loss'])
    ctc_loss.extend(train_stat['ctc_loss'])



hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']
char_set_path = hw_network_config['char_set_path']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k,v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v
    
config["network"]["hw"]["num_of_outputs"] = len(idx_to_char) + 1
training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = HwDataset(training_set_list,
                          char_set['char_to_idx'], augmentation=True,
                          img_height=hw_network_config['input_height'], 
                          flip_image=False)

train_dataloader = DataLoader(train_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=True, num_workers=0, drop_last=True,
                             collate_fn=hw_dataset.collate)
print('Train dataloader len: ', len(train_dataloader.dataset))

batches_per_epoch = int(pretrain_config['hw']['images_per_epoch']/pretrain_config['hw']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = HwDataset(test_set_list,
                         char_set['char_to_idx'],
                         img_height=hw_network_config['input_height'], 
                         flip_image = False)

test_dataloader = DataLoader(test_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=False, num_workers=0,
                             collate_fn=hw_dataset.collate)
print('Test dataloader len: ', len(test_dataloader.dataset))

criterion = CTCLoss(blank=0, zero_infinity=True)



if TRANSFER_HW:    
    print('Transfer learning')
    hw = create_hw_from_pretrained(hw_network_config['transfer_path'], hw_network_config, len(idx_to_char) + 1)
else:
    print('creating network')
    hw = cnn_lstm.create_model(hw_network_config)
    

if LOAD_HW:
    print('loading from saved')
    hw_path = os.path.join(pretrain_config['snapshot_path'], "hw_last.pt")
    hw_state = safe_load.torch_state(hw_path)
    hw.load_state_dict(hw_state)

DEVICE = 'cuda'
hw.to(DEVICE)

optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
dtype = torch.cuda.FloatTensor

lowest_loss = get_test_loss()
cnt_since_last_improvement = 0
# This will stop when time limit is reached
# Give a smaller range if it is to terminate on number of epochs
for epoch in range(1000000):
    first = True
    print ("Epoch", epoch)
    
    hw.train()
    total_ctc_loss = 0.0
    count = 0
    for i, x in enumerate(train_dataloader):
        
        
        line_imgs = x['line_imgs'].to(DEVICE)
        labels =  x['labels']
        label_lengths = x['label_lengths']
        preds = hw(line_imgs).cpu()
        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()
        batch_size = preds.size(1)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, labels, preds_size, label_lengths)
        total_ctc_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = count + 1
        
    print('....ctc loss', total_ctc_loss)
    
    
    print( "Real Epoch", train_dataloader.epoch)
    ctc_loss.append(total_ctc_loss.item())
    model_path = pretrain_config['snapshot_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(hw.state_dict(), os.path.join(model_path, 'hw_last.pt'))
    # Save training history
    train_stat = {'ctc_loss': ctc_loss, 'valid_loss': valid_loss}
    log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=True)    
    hw_test_loss = get_test_loss()

    print("Test Loss", hw_test_loss, lowest_loss)
    valid_loss.append(hw_test_loss)
    
    if lowest_loss > hw_test_loss:

        lowest_loss = hw_test_loss
        print ("Saving Best")
        torch.save(hw.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'hw.pt'))

    # This will break the loop    
    if time.time() - init_training_time > allowed_time:
        break        

    sys.stdout.flush()
    sys.stderr.flush()

train_stat = {'ctc_loss': ctc_loss, 'valid_loss': valid_loss}
log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=False)    

print('Done ...')
sys.stdout = old_stdout
sys.stderr = old_stderr
print('Done ...')


