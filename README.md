# VRImitationLearning

## About the Project
This is a partial implementation of the paper [Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation](https://arxiv.org/abs/1710.04615) written for Danfei Xu's Spring 2023 class CS 8803: Deep Learning for Robotics at Georgia Tech.  More info about this class can be found [here](https://sites.google.com/view/gt-cs-8803-dlr/home).

A google slides presentation about the project can be found [here](https://docs.google.com/presentation/d/155gPehIv7s59MdSMAwriqRt6pkf0tdznTQnVsFF27Qg/edit?usp=sharing).

## Project Functionality
This repo has 3 functions.
1) Use virtual reality to collect training data of a simulated pybullet environment.  [Video of Data Collection](https://www.youtube.com/watch?v=3JOXGN6TEGQ)
2) Use this training data to train the model described in the above paper.
3) Run this trained model in the simulated enviornment to validate performance. [Video of Model Running](https://www.youtube.com/watch?v=n4WgxOyaxc4)

## Project Files
**datacollector.ipynb** - a notebook for generating training data and running a pretrained model.

**trainer.ipynb** - a notebook for training the model with collected data.

**VRNet.py** - a pytorch implementation of the imitation learning model described in the above paper.  Note that this contains a SpatialSoftmax implementation from [here](https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834).

**model.pt** - a pretrained model which can place the cube on the tray after starting on the cube. 

## Example Images from training set (160x120 pixel rgb and depth images)

RGB Image:

![RGB Image](https://github.com/NathanMalta/VRImitationLearning/blob/main/rgbImg.png)

Depth Image:

![Depth Image](https://github.com/NathanMalta/VRImitationLearning/blob/main/depthImg.png)

The full training set is available upon request.  If there's interest I can upload it to google drive or kaggle.
