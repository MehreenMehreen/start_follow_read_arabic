# Inference Results
## Cross-Dataset validation results of SFR-Arabic

The line-level cross dataset evaluation of different training and test sets has the following CER:

|                 | Test    | Test    | Test    |
|                 | MADCAT  |  KHATT  | Muharaf |
|:---------------:|--------:|:-------:|:-------:|
| Train MADCAT    |  0.05   | 0.178   |  0.435  |
| Train KHATT     |  0.170  | 0.142   |  0.438  |
| Train Muharaf   |  0.277  | 0.331   |  0.148  |     


### Some notes on the above results
- All Muharaf images were used for testing
- KHATT was trained and tested on the split defined by authors of KHATT
  - There were some images that we cropped before training and testing. If an image height was more than 200 we cropped it to 200.
- We defined our own split for training, validation, and test sets of Muharaf
- When Muharaf was trained, we defined 3 different splits for training, validation, and testing.
  - The individual result for each split when tested on Muharaf is [0.14316096196812786, 0.15020022284820123, 0.1520919499638009]  
  - The individual result for each split when tested on MADCAT is [0.30519922011142503, 0.25094161618281424, 0.27520163281485976]
  - The individual result for each Muharaf split when tested on KHATT is [0.3845364978277071, 0.28704252327838536, 0.3223472860715881]






