Trying to benchmark all the optimizers used in deep learning


## Network
conv1 - 64 - 5*5 - 1*1  
maxpool2 - 3*3, 2*2  
local_response_normalization  
conv2 - 64 - 5*5 - 1*1  
maxpool2 - 3*3, 2*2  
local_response_normalization  
FC - 384  
FC - 192  
FC - Final (Mnist , CIFAR10 - 10, CIFAR100 - 100)  


## Optimizers to test
- adadelta
- adagrad
- adam
- ftrl
- momentum
- rmsprop
- sgd

## Things to check
- learning_rate behaviour with-in each optimizer
- time taken to run each optimizer
- hyper-parameters to tune (check for one)
- maximum training, testing and validation accuracy achieved
- graphs

## Datasets
- MNIST
- CIFAR10
- CIFAR100



The models are trained with
batch_size: 32
epochs: 100

train, validation and test accuracies are obtained after the 100th epoch.

#Experiments

## 1. Tensorflow default values:
 - We have used a learning_rate of 0.001 for all the optimizers keeping all other params constant
 - Momentum=0.9 in case of momentum optimizer
 
## MNIST Dataset
| optimizer        |train_accuracy   | Validation_Accuracy  | Test Accuracy |
| -------------    |:-------------:  | --------------------:| --------------|
| adadelta         |    00.00%       |       00.00%         |    00.00%     |
| adagrad          |    00.00%       |       00.00%         |    00.00%     |
| adam             |    00.00%       |       00.00%         |    00.00%     |
| ftrl             |    00.00%       |       00.00%         |    00.00%     |
| momentum         |    00.00%       |       00.00%         |    00.00%     |
| rmsprop          |    00.00%       |       00.00%         |    00.00%     |
| sgd              |    00.00%       |       00.00%         |    00.00%     |

## CIFAR10 Dataset
| optimizer        |train_accuracy   | Validation_Accuracy  | Test Accuracy |
| -------------    |:-------------:  | --------------------:| --------------|
| adadelta         |    00.00%       |       00.00%         |    00.00%     |
| adagrad          |    00.00%       |       00.00%         |    00.00%     |
| adam             |    00.00%       |       00.00%         |    00.00%     |
| ftrl             |    00.00%       |       00.00%         |    00.00%     |
| momentum         |    00.00%       |       00.00%         |    00.00%     |
| rmsprop          |    00.00%       |       00.00%         |    00.00%     |
| sgd              |    00.00%       |       00.00%         |    00.00%     |


## CIFAR100 Dataset
| optimizer        |train_accuracy   | Validation_Accuracy  | Test Accuracy |
| -------------    |:-------------:  | --------------------:| --------------|
| adadelta         |    00.00%       |       00.00%         |    00.00%     |
| adagrad          |    00.00%       |       00.00%         |    00.00%     |
| adam             |    00.00%       |       00.00%         |    00.00%     |
| ftrl             |    00.00%       |       00.00%         |    00.00%     |
| momentum         |    00.00%       |       00.00%         |    00.00%     |
| rmsprop          |    00.00%       |       00.00%         |    00.00%     |
| sgd              |    00.00%       |       00.00%         |    00.00%     |
