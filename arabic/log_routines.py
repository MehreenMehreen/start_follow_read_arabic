import csv
import os
import sys

# Set the max size for csv fields
csv.field_size_limit(sys.maxsize)
def save_log(epochs, train_loss, test_loss, log_file, best=True, overwrite=True):
    loss_dict = {'epochs': epochs, 'train_loss': train_loss, 'valid_loss': test_loss, 
                 'best': best}
    mode = 'w' if overwrite else 'a'
    with open(log_file, mode) as outfile:
        strm = csv.writer(outfile)
        for key, val in loss_dict.items():
            strm.writerow([key, val])

def read_log(log_file):
    with open(log_file, 'r') as fstream:
        reader = csv.reader(fstream)    
        d = dict(reader)
    return eval(d['epochs']), eval(d['train_loss']), eval(d['valid_loss'])


def save_log_dict(dict_to_save, log_file, overwrite):
    mode = 'w' if overwrite else 'a'
    with open(log_file, mode) as outf:
        strm = csv.writer(outf)
        for key, val in dict_to_save.items():
            strm.writerow([key, val])
            
def read_log_dict(log_file):
    if not os.path.exists(log_file):
        return None
    with open(log_file, 'r') as fin:
        reader = csv.reader(fin)
        dict_log = dict(reader)
    for key, val in dict_log.items():
        dict_log[key] = eval(val)
    return dict_log


def get_std_dump_filename(config_file, model):
    model_filename = config_file.replace('config', 'dump_{}_'.format(model))
    return model_filename
   
        