# Inference Results
## Cross-Dataset validation results of SFR-Arabic

              TEST
| Train\Test      | MADCAT  |  KHATT  | Muharaf |
|:---------------:|--------:|:-------:|:-------:|
| Train MADCAT    |         | 0.178   |  0.435  |
| Train KHATT     |  0.170  |         | 0.438   |
| Train Muharaf   |  0.277  | 0.331   |         |


### Some notes on the above results
- All Muharaf images were used for testing
- KHATT was trained and tested on the split defined by authors of KHATT
- We defined our own split for training, validation, and test sets of Muharaf
- When Muharaf was trained, we defined 3 different splits for training, validation, and testing. The individual result for each split when tested on MADCAT is [0.30519922011142503, 0.25094161618281424, 0.27520163281485976]
- The individual result for each Muharaf split when tested on KHATT is [0.3845364978277071, 0.28704252327838536, 0.3223472860715881] 





