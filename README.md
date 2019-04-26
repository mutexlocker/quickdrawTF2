# Quick, Draw! 
This repository contains code for training and inference different neural networks on Google's Quick, Draw! Dataset. 
## Dataset 
Download numpy bitmap files to the data directory. These files contain 28*28 image data of doodle drawings. 
```
cd logs/
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy .
```
## Networks
Implementation has multiple networks. The best results were achieved by *cnn_model3* with top5 category accuracy of 95%. 
## Config
By default, the code trains for 100 classes and loads 15000 samples for each class.
You can change the number of classes to train by changing the value of *totalclasses* and number of samples for each class by changing the value of  *samples*
## Execution 
By default the code runs *cnn_model3* but can be changed by this line of code:
```
model_cnn,tensorboard = cnn_model3()
```
chnage *cnn_model3()* to any other networks in the list:
- *cnn_model_leaky()* CNN with leaky RELU activation
- *cnn_model_lstm()* LSTM Network 


## Logging
Tensoboard is also supported. logs are writtein in *logs* folder. Run tensoboard as follows:
```
tensorboard --logdir logs
```

## Save Network 
The trained neural network will be saved as two separate file (model.h5 and model.json) when training is completed.
