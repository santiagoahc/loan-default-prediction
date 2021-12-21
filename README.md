# loan-default-prediction

## Modelling process

### 1. Data Preparation
The train data has null values that were filled with the mean. The columns 
with object type were removed. 

Then in order to understand the data, we analyse the correlations between the 
column features. Finally, columns with a correlation higher than a thresholds 
are removed for the sake of removing redundant information, helping the training 
process, using less but meaningful data to train. It is also improve the 
model interpretability. Finally, the data was standardized.

### 2. Training and test prediction
The goal of the model is to predict the loss is case of default, so if not 
default the loss is 0 and in case of full default the loss is 100.
We train a xgboost regressor on the full train set and then predict on the 
test set. For the sake of limiting the task to 3 hours we haven't done 
parameter tuning with GridSearch for example.

### MAE score
The MAE score is: 