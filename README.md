# Interaction-Aware Sampling-Based MPC with Learned Local Goal Predictions

<img src="docs/imgs/framework.png">

This repository contains the training and testing scripts for the prediction model used in the "Interaction-Aware Sampling-Based MPC with Learned Local Goal Predictions" paper.

### Data

To download the data and trained models (prediction model and pre-trained autoencoder), run
```
./download_data.sh
``` 

### Setup

This repository requires python <= 3.7 and tensorflow == 1.15.x. The instructions were tested on Ubuntu 16.

```
./install.sh
source social_vdgnn/bin/activate
```

### Training 

The training script for the model is src/train_roboat.py. In order to train a new model on the data, specify a model number (--exp_num) and the training parameters in /src/train.sh. An example is provided in the file. Then in /src, run 
```
./train.sh
```

### Testing

In a similar fashion, specify training parameters for the trained model in /src/test.sh (examples provided) and run
```
./test.sh
```

Model parameters will be automatically saved in /trained_models
 

### If you find this code useful, please consider citing:

```
@inproceedings{
    ...
}
```