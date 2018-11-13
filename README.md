# houseNet

## Overview
This neural network project was created to evaluate the effect of dilated convolutions on a semantic segmentation task.
The dataset from the [crowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)
will be used to test dilated convolutions for binary segmentation. 

## Usage
The neural network for this project was programmed by using [Keras](https://keras.io/) with the [Tensorflow](https://www.tensorflow.org/) backend.
The project code should be usable if you have installed Keras and Tensorflow.
Keep in mind that this project was designed with the GPU version of Tensorflow and training will be slow or may not work at all with your CPU.

The train.py file can be used to start training if you have a suitable dataset available and created a configuration file.

## Dataset
Current testings are done on down-sampled versions of the images from the mapping challenge.
The images were converted to [NumPy](http://www.numpy.org/) arrays with the shape (150, 150, 3) and stored in single .npy files.
During training they are loaded in batches and randomly flipped or rotated.

The code for generating such a dataset will be published in the future.

## Configuration
For loading the training/validation data a .ini file is read.
The file should contain the following content:

```ini
[DIRECTORIES]
train_data  = /path/to/data/train/
val_data    = /path/to/data/val/
models      = /path/to/models/
logs        = /path/to/logs/
```

## Current state
This project is currently "work in progress" and will get some updates in the future.
However many tasks have already been completed and the work can now be focused on testing
different network architectures (especially using dilated convolutions).