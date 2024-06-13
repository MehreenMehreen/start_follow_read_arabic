# Start, Follow, Read &mdash; Arabic
The code in this repository is modified from the source code: [Start, Follow, Read](https://github.com/cwig/start_follow_read) and its [Python3 version](https://github.com/sharmaannapurna/start_follow_read_py3). See their detailed paper:.
[Curtis Wigington, Chris Tensmeyer, Brian Davis, William Barrett, Brian Price, and Scott Cohen. Start, Follow, Read: End-to-end full-page handwriting recognition. In Proceedings of the European Conference on Computer Vision (ECCV), pages 372–388, 2018.](https://openaccess.thecvf.com/content_ECCV_2018/html/Curtis_Wigington_Start_Follow_Read_ECCV_2018_paper.html)

We release this code under the same license as given as given on this page [Start, Follow, Read](https://github.com/cwig/start_follow_read).

You can use this code to train and test handwriting recognition (HTR) of Arabic historic manuscripts.

## Dataset
For the dataset, please see [Muharaf: Manuscripts of handwritten Arabic for cursive text recognition](https://github.com/MehreenMehreen/muharaf). 

## Step 0: Setting up
1. Clone the repo
2. Setup the environment
3. Download the trials folders. These are required for inference on your own handwritten Arabic pages. These are also required for reproducing the results of our paper.



## Step 1: Preprocessing
The start, follow, read (SFR) system requires files in a specific format for training. You can download the preprocessed files for [Muharaf-public](https://zenodo.org/records/11492215). Make sure to extract them in the `main start_follow_read_arabic` directory.

In order to preprocess the files yourself, follow these steps:
1. Download the Muharaf-public dataset and place it in `data_files/public` folder in the `start_follow_read_arabic` directory.
2. On the command line, switch to the main `start_follow_read_arabic` directory. Activate the sfr environment.
3. From the command line run:
   ```
   python arabic/preprocessing_all.py ../data_files/public
   ```
Running this will create the following directory structure. You can ignore the restricted part, if you don't have these files. The system can be trained and tested witht the Muharaf-public part only:   

### Directory structure
The preprocessed directory looks like this: ![directory structure](images/directory_structure.png)

## Step 2: Splitting data into (Train, Validate, Test) sets
1. We recommend that you create a directory `trials` in main start_follow_read_arabic directory. Within this directory, you can run various trials.
2. For example, we ran a trial on Muharaf-public with a split of (1100, 50, 66) images. This trial consisted of three different random splits of (train, validate, test) sets. [Download this folder here](https://zenodo.org/records/11492215). The three sets are in trials->public_1100 folder in sub-folders `set0`, `set1`, `set2`.
3. To create random splits for a single trial run the script create_trials in the arabic directory. In the arguments specify:
   1. input_dir: The input directory for preprocessed files. Default is data_files/sfr_arabic
   2. input_files: The list of input files created after preprocessing, e.g., ['public_sfr'] (default) for only public files or \['public_sfr', 'restricted_sfr'\]. These are the names of json files containing \[json image\] pairs (without the '.json' extension) in the sfr_arabic directory.
   3. train_valid: The total number of files in (train, valid) sets. The rest will go in the test set. Default for our public_1100 trial is 1100 50.
   4. output_dir: The output directory (default is trials/public_1100)
   5. total_sets: The total number of sets (default 3). It will create the sub-folders `set0`, `set1`, `set2`. For the public trial, each set folder has `pretrain_train_1150.json`, `pretrain_valid_1150.json`, `pretrain_test_1150.json` files containing the list of [json image] pairs. It also has a config_1150.yaml file. Note 1150 is total_train+total_valid examples.
   6.  To reproduce the results of our paper on the public part of Muharaf, create the trial directory by running:
      ```
      python arabic/create_trials.py
      ```
  
  

## Step 3: Training
This system only uses stage 1 (pretraining stage) training of SFR. You can train all networks in parallel or train them one by one.
### SOL Network
use the script arabic/stage1_sol.py with the configuration file as argument. For example to train on trials/public_1100/set0:
```
python arabic/stage1_sol.py trials/public_1100/set0/config_1150.yaml
```

### LF Network
use the script arabic/stage1_lf.py with the configuration file as argument. For example to train on trials/public_1100/set0:
```
python arabic/stage1_lf.py trials/public_1100/set0/config_1150.yaml
```

### HW Network
use the script arabic/stage1_hw.py with the configuration file as argument. For example to train on trials/public_1100/set0:
```
python arabic/stage1_hw.py trials/public_1100/set0/config_1150.yaml
```

## Step 4: Inference
### Results on the page images of test set 
To run inference on test images for a particular trial, use the script `page_predictions.py`. To look at various command line arguments for this script, type:
```
python arabic/page_predictions.py -h
```
For example, to run inference on all test files (filename read from configuration file) in a trial directory, run the following: 
```
python arabic/page_predictions.py --main_dir trials/public_1100 --train_valid 1150
```
This script will output the CER and WER on the entire page image.
### Results on the line images of test set 
Run `arabic/line_predictions.py` to get the performance of the HW network on line images (without segmenting the page). It takes three arguments:
1. Argument 1 is the name of the trial directory
2. Total sets in the trial directory
3. Name of the configuration file
For example to get the line predictions on the `public_1100` trial folder type:
```
python arabic/line_predictions.py trials/public_1100 3 config_1150.yaml 
```
Running the above will reproduce the results of the paper
### Inference on a handwritten page image
Run inference on your own handwritten Arabic page images using `arabic/annotate_files_in_directory.py`. This script will read all the image files (jpg extension) in a directory and create a corresponding JSON file for each image. For example, to get the predictions in a directory called images, run the following. Here the second argument specifies the configuration file to use. We recommend using `trial_15` `set0` config file as it has the best CER/WER results (compared to `set1` and `set2`):
python arabic/annotate_files_in_directory.py images/ trials/trial_15/set0/config_1550.yaml ```
```
