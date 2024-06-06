import sys
sys.path.append('py3/')

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lf
from lf import lf_dataset, lf_loss
from lf.lf_dataset import LfDataset
from lf.line_follower import LineFollower
from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import numpy as np
import cv2
import sys
import json
import os
import yaml
import csv
from utils import safe_load
import time

import log_routines


def get_lf_test_loss():
    sum_loss = 0.0
    steps = 0.0
    line_follower.eval()
    for x in test_dataloader:
        x = x[0]

        positions = [Variable(x_i.type(dtype), requires_grad=False, volatile=True)[None,...] for x_i in x['lf_xyrs']]
        xy_positions = [Variable(x_i.type(dtype), requires_grad=False, volatile=True)[None,...] for x_i in x['lf_xyxy']]
        img = Variable(x['img'].type(dtype), requires_grad=False, volatile=True)[None,...]

        if len(xy_positions) <= 1:
            continue

        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions), skip_grid=True)
        loss = lf_loss.point_loss(xy_output, xy_positions)
        sum_loss += loss.data.item() #[0]
        steps += 1
    return sum_loss/steps

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    old_stdout = sys.stdout
    dump_file = log_routines.get_std_dump_filename(config_file, "pretrain_lf")
    #sys.stdout = open(dump_file, 'w')   

    old_stderr = sys.stderr
    dump_file = log_routines.get_std_dump_filename(config_file, "pretrain_lf_err")
    #sys.stderr = open(dump_file, 'w')   
    

init_training_time = time.time()
allowed_time = config['training']['lf']['reset_interval']    

train_loss = []
valid_loss = []

LOG_FILE = config['pretraining']['lf']['log_file']
train_stat = log_routines.read_log_dict(LOG_FILE)
if train_stat is not None:
    train_loss.extend(train_stat['train_loss'])
    valid_loss.extend(train_stat['valid_loss'])
LOAD_LF = False
if os.path.exists(config['pretraining']['snapshot_path']+'/lf.pt'):
    LOAD_LF = True
else:
    print('train from scratch')

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

training_set_list = load_file_list(pretrain_config['training_set'])

train_dataset = LfDataset(training_set_list,
                          augmentation=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True, num_workers=0,
                              collate_fn=lf_dataset.collate)
print('Train dataloader len: ', len(train_dataloader.dataset))

batches_per_epoch = int(pretrain_config['lf']['images_per_epoch']/pretrain_config['lf']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = LfDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=lf_dataset.collate)
print('Test dataloader len: ', len(test_dataloader.dataset))

line_follower = LineFollower()
if LOAD_LF:
    print('loading from file')
    lf_state = safe_load.torch_state(os.path.join(pretrain_config['snapshot_path'], "lf_last.pt"))
    # special case for backward support of
    # previous way to save the LF weights
    if 'cnn' in lf_state:
        new_state = {}
        for k, v in lf_state.items():
            if k == 'cnn':
                for k2, v2 in v.items():
                    new_state[k+"."+k2]=v2
            if k == 'position_linear':
                for k2, v2 in  v.state_dict().items():
                    new_state[k+"."+k2]=v2
            # if k == 'learned_window':
            #     new_state[k]=nn.Parameter(v.data)
        lf_state = new_state
    
    
    line_follower.load_state_dict(lf_state)
     
line_follower.cuda()
optimizer = torch.optim.Adam(line_follower.parameters(), 
                             lr=pretrain_config['lf']['learning_rate'])

dtype = torch.cuda.FloatTensor

lowest_loss = get_lf_test_loss()

cnt_since_last_improvement = 0

# Will end when time limit is reached
for epoch in range(100000000):
    print ("Epoch", epoch)
    sum_loss = 0.0
    steps = 0.0
    line_follower.train()
    for x in train_dataloader:
        #Only single batch for now
        x = x[0]

        positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyrs']]
        xy_positions = [Variable(x_i.type(dtype), requires_grad=False)[None,...] for x_i in x['lf_xyxy']]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None,...]

        if len(xy_positions) <= 1:
            continue

        reset_interval = 4
        grid_line, _, _, xy_output = line_follower(img, positions[:1], steps=len(positions), all_positions=positions,
                                           reset_interval=reset_interval, randomize=True, skip_grid=True)

        loss = lf_loss.point_loss(xy_output, xy_positions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.data.item()
        steps += 1

    print("Train Loss", sum_loss/steps)
    print("Real Epoch", train_dataloader.epoch)
    model_path = pretrain_config['snapshot_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(line_follower.state_dict(), os.path.join(model_path, 'lf_last.pt'))
            # Save training history
    train_stat = {'train_loss': train_loss, 'valid_loss': valid_loss}
    log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=True)  

    train_loss.append(sum_loss/steps)

    lf_test_loss = get_lf_test_loss()

    cnt_since_last_improvement += 1
    print("Test Loss", lf_test_loss, lowest_loss)
    valid_loss.append(lf_test_loss)
    
    if lowest_loss > lf_test_loss:
        cnt_since_last_improvement = 0
        lowest_loss = lf_test_loss
        print("Saving Best")
        torch.save(line_follower.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'lf.pt'))
 
    if time.time() - init_training_time > allowed_time:
        break


train_stat = {'train_loss': train_loss, 'valid_loss': valid_loss}
log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=False)  
print('Done')  

sys.stdout = old_stdout
sys.stderr = old_stderr
print('Done')
