import sys
sys.path.append('py3/')

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sol
from sol.start_of_line_finder import StartOfLineFinder
from sol.alignment_loss import alignment_loss
from sol.sol_dataset import SolDataset
from sol.crop_transform import CropTransform

from utils.dataset_wrapper import DatasetWrapper
from utils.dataset_parse import load_file_list

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import cv2
import json
import yaml
import sys
import os
import math
import time
from utils import transformation_utils, drawing
from utils import safe_load
import csv

import log_routines




if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    old_stdout = sys.stdout
    dump_file = log_routines.get_std_dump_filename(config_file, "pretrain_sol")
    #sys.stdout = open(dump_file, 'w')       

    old_stderr = sys.stderr
    dump_file = log_routines.get_std_dump_filename(config_file, "pretrain_sol_err")
    #sys.stderr = open(dump_file, 'w')     

def get_sol_test_loss():    
    sol.eval()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(test_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False, volatile=True)
        sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False, volatile=True)

        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)
        sum_loss += loss.item()
        steps += 1
    return sum_loss/steps

    
LOAD_SOL = False
if os.path.exists(config['pretraining']['snapshot_path']+'/sol.pt'):
    LOAD_SOL = True
else:
    print('train from scratch')
LOG_FILE = config['pretraining']['sol']['log_file']

train_loss = []
valid_loss = []    

train_stat = log_routines.read_log_dict(LOG_FILE)
if train_stat is not None:
    train_loss.extend(train_stat['train_loss'])
    valid_loss.extend(train_stat['valid_loss'])
    
init_training_time = time.time()
allowed_time = config['training']['sol']['reset_interval']

sol_network_config = config['network']['sol']
pretrain_config = config['pretraining']

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = SolDataset(training_set_list,
                           rescale_range=pretrain_config['sol']['training_rescale_range'],
                           transform=CropTransform(pretrain_config['sol']['crop_params']))

train_dataloader = DataLoader(train_dataset,
                              batch_size=pretrain_config['sol']['batch_size'],
                              shuffle=True, num_workers=0,
                              collate_fn=sol.sol_dataset.collate)

print('Train dataloader len: ', len(train_dataloader.dataset))

batches_per_epoch = int(pretrain_config['sol']['images_per_epoch']/pretrain_config['sol']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = SolDataset(test_set_list,
                          rescale_range=pretrain_config['sol']['validation_rescale_range'],
                          transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=sol.sol_dataset.collate)
print('Test dataloader len: ', len(test_dataloader.dataset))


base0 = sol_network_config['base0']
base1 = sol_network_config['base1']
sol = StartOfLineFinder(base0, base1)

if LOAD_SOL:    
    print('loading from file')
    sol_state = safe_load.torch_state(os.path.join(pretrain_config['snapshot_path'], "sol_last.pt"))
    sol.load_state_dict(sol_state)

if torch.cuda.is_available():
    sol.cuda()
    dtype = torch.cuda.FloatTensor
else:
    print ("Warning: Not using a GPU, untested")
    dtype = torch.FloatTensor

alpha_alignment = pretrain_config['sol']['alpha_alignment']
alpha_backprop = pretrain_config['sol']['alpha_backprop']

optimizer = torch.optim.Adam(sol.parameters(),
                             lr=pretrain_config['sol']['learning_rate'])


lowest_loss = np.inf
cnt_since_last_improvement = 0

# Get lowest accuracy
lowest_losss = get_sol_test_loss()


for epoch in range(1000000000):
    print("Epoch", epoch)

    sol.train()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(train_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False)

        sol_gt = None
        if x['sol_gt'] is not None:
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)


        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        steps += 1

    print ("Train Loss", sum_loss/steps)
    print ("Real Epoch", train_dataloader.epoch)
    model_path = pretrain_config['snapshot_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(sol.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol_last.pt'))
    train_stat = {'train_loss': train_loss, 'valid_loss': valid_loss}
    log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=True) 
    train_loss.append(sum_loss/steps)
    
    sol_test_loss = get_sol_test_loss()

    cnt_since_last_improvement += 1
    if lowest_loss > sol_test_loss:
        cnt_since_last_improvement = 0
        lowest_loss = sol_test_loss
        print("Saving Best")

        torch.save(sol.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'sol.pt'))
        
         

    print ("Test Loss", sol_test_loss, lowest_loss)


    valid_loss.append(sol_test_loss)
    
    if time.time() - init_training_time > allowed_time:
        break
      


train_stat = {'train_loss': train_loss, 'valid_loss': valid_loss}
log_routines.save_log_dict(train_stat, LOG_FILE, overwrite=False)  

print('Done')  
sys.stdout = old_stdout
sys.stderr = old_stderr
print('Done')
