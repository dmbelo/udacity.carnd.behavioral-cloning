# Network Architecture Overview

# Sample Usage

To use this model first launch the Udacity Self Driving Car Nanodegree Simulator and enter "Autonomous Mode". Once the simulator is live, launch the trained model by running `python drive.py model.json` and watch it take control of the vehicle.

# Model Architecture

[INSERT GRAPHIC OF LAYERS]
![Model Architecture](etc/ModelArchitecture.png)

The model architecture implemented has been largely based on the NVIDIA model presented in their their end-to-end approach paper



[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)

This model uses 4  convolutional layers followed by 4 fully connected layers.

# Training Data

[INSERT GRAPHIC OF DISTRIBUTION]
[INSERT GRAPHIC OF TIME SERIES]

The training data used for the model was derived the data set provided by the Udacity which consisted of 8036 images with their respective steering, braking controls and vehicle velocity. In addition left and right camera angles were provided to compliment each center image.

The process of generating the training data from this set was a critical step of the learning approach and it was divided into two steps: pre-process and real-time data augmentation.

The pre-process step was primarily

The real-time data augmentation step consisted of
