# Interaction-Aware Sampling-Based MPC with Learned Local Goal Predictions

<img src="docs/imgs/framework.png">

This repository contains the training and testing scripts for the prediction model used in the "Interaction-Aware Sampling-Based MPC with Learned Local Goal Predictions" paper.

### Data

Download the dataset here: <a href="https://www.surfdrive.surf.nl/files/index.php/s/2x9P82EOjNWDVwP">Dataset</a>
Move the data folder into the repository.

### Setup

This repository requires python <= 3.7 and tensorflow == 1.15.x. The instructions were tested on Ubuntu 16.

```
./install.sh
source social_vdgnn/bin/activate
./train.sh
./test.sh
```


### If you find this code useful, please consider citing:

```
@inproceedings{
    ...
}
```