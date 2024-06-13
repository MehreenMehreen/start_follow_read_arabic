import sys
sys.path.append('py3/')
sys.path.append('arabic/')
import os
import test_hw_helper_routines as test_hw
import numpy as np

from tokenizers import Tokenizer
from utils import error_rates
# Sets is a list of set directory names in target_dir. config_file has the same name in all directories
all_results_df = []
def get_predictions(sets, target_dir, config_to_use, test_file="", result_file="", 
                    pt_file="hw.pt", tokenizer=None):
    mean_cer_list = []
    mean_wer_list = []
    for s in sets:
        main_dir = target_dir + s
        
        print('doing:', config_to_use, 'in', main_dir)        
        config_file = os.path.join(main_dir, config_to_use)
        if test_file != "":
            test_file_to_use = os.path.join(main_dir, test_file)
        else:
            test_file_to_use = ""
        if len(result_file) == 0:
            result_file = ""
        else:
            result_file = os.path.join(main_dir, result_file)
        
        result_df = test_hw.write_csv_all_predictions(config_file, pt_file=pt_file, 
                                                      test_file_to_use=test_file_to_use, 
                                                      result_file=result_file, tokenizer=tokenizer)
        
        mean_cer = result_df['CER'].mean()
        #mean_image_wise_cer = compute_file_wise_cer
        print(f'mean_cer for {s}', mean_cer)
        mean_wer = result_df['WER'].mean()
        print(f'mean_wer for {s}', mean_wer)
        mean_cer_list.append(mean_cer)
        all_results_df.append(result_df)
        mean_wer_list.append(mean_wer)
    return all_results_df, mean_cer_list, mean_wer_list

if __name__ == "__main__":
    target_dir = 'trials/public_1100/'
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    if target_dir[-1] != '/':
        target_dir += '/'
    
        
    total_sets = 3
    if len(sys.argv) > 2:
        total_sets = int(sys.argv[2])
     
    sets = [f'set{n}' for n in range(total_sets)]
    
    config_file = 'config_1150.yaml'
    if len(sys.argv) > 3:
        config_file = sys.argv[3]
        
    all_results_df, mean_cer_list, mean_wer_list = get_predictions(sets, 
                                                                   target_dir, 
                                                                   config_file)


    print('CER list', mean_cer_list)
    print('WER list', mean_wer_list)
    print('Mean CER: {:.4f}'.format(np.mean(mean_cer_list)))

    print(f'STD CER: {np.std(mean_cer_list)}')

    print('Mean WER: {:.4f}'.format(np.mean(mean_wer_list)))

    print(f'STD WER: {np.std(mean_wer_list)}')
    print('done')
