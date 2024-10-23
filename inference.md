# Inference Results
## Cross-Dataset Validation Results

              TEST
| Train\Test      | MADCAT  |  KHATT  | Muharaf |
|:---------------:|--------:|:-------:|:-------:|
| Train MADCAT    |         | 0.178   |  0.435  |
| Train KHATT     |  0.170  |         | 0.438   |
| Train Muharaf   |  0.277  | 0.331   |         |


Train MADCAT (using train pickle files on OIT research drive), Test Muharaf (all of it): 0.43525671302561875
Train MADCAT (using train pickle files on OIT research drive), Test KHATT (test split of authors): 0.17846951774230058


Train KHATT (train set of KHATT creators), Test MADCAT (using test pickle files on OIT research drive): 0.16974063275934687
Train KHATT (train set of KHATT creators), Test Muharaf (all of it): 0.4381161194809963

Train Muharaf (use all three train splits, and average), Test KHATT (test set of creators): 
[0.3845364978277071, 0.28704252327838536, 0.3223472860715881]  average all three: 0.33130876905922685

Train Muharaf (use all three train splits, and average), Test MADCAT (using test pickle files on OIT research drive): 
[0.30519922011142503, 0.25094161618281424, 0.27520163281485976] average all three: 0.27711415636969966

