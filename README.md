# udacity.carnd.behavioral-cloning

# Sample Usage

To run this model first launch the Udacity Self Driving Car Nanodegree Simulator and enter "Autonomous Mode". Once the simulator is live, launch the trained model by running `python drive.py model.json` and watch as it takes control of the vehicle!

# Model Architecture Overview

The implemented model architecture is largely based on the NVIDIA CNN model: [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf).

![Model Architecture](etc/ModelArchitecture.png)

The model uses 5 convolutional layers followed by 4 fully connected layers. All layers feature a rectifier linear unit (ReLU) activation layer on their output which introduce non-linearities into the model.

Furthermore, all convolutional layers include some form of down sampling: a stride greater than 1 is used for the first 3 layers, and all layers use a valid padding. These value have been chosen to improve the training speed while balancing minimal impact on performance observed on the simulator.

No dropouts layers are included in the original NVIDIA model and none are introduced here. Dropouts are typically included to minimize over-fitting but throughout the design and training process it seemed that over-fitting minimization techniques implemented in the data augmentation step were satisfactory on their own.

In total the model has 8,002,167 parameters which are learned through the training process.

# Training Data

The training data used for the model was derived from the data set provided by Udacity. This data set consists of 8036 images of simulator driving. Along with the images, their respective steering, braking control and vehicle velocity values were provided. Finally, left and right camera angle perspectives were also included to compliment each center image (for a total of 24,108 images).

The process of generating the training data from this set is the critical step in the learning approach. The approach is divided into two steps: pre-process and real-time augmentation.

## Pre-process

The pre-process step primarily deals with removing images that were deemed unnecessary or misleading. The data set contains initial and final sections which are primarily ramping up and slowing down and not actually representative of normal driving conditions. This can be seen in the erratic steering, and speed trace shown below. Around 3500 images in, there is also another erratic driving period which is not smooth or representative of normal driving. For these reasons, the three sections of data, highlighted in red below, are removed from the data set.

![Vehicle Data](etc/vehicle_signals.png)

The original data set is also highly biased towards straight line driving. This is shown below in the left histogram. By selecting a random set of 500 images from the subset of zero-steer images, the distribution was made to be much more representative of a normal distribution, shown on the right histogram. This results in a more balanced data set which does not bias the model toward a propensity to drive straight as much as the original set.

![Histogram](etc/hist.png)

The final pre-process step was to assign a steering angle to the left and right camera angle perspectives and use those images in the training set. Besides tripling the amount of data available it also provides essential data to 'show' the model to recover from driving close to the edge of the road without actually driving close to the edge. To synthesize the steering labels for these images, an offset of 0.2 was added/subtracted to the center steer value and assigned to the left and right images. This value was chosen arbitrarily and tuned in various model training iterations.

The final set of images in the data set consists of 10,725 (the original set was 24,108). Also of note is that while the original images are of size 320x160, these were cropped and scaled down to a size of 43x160. Cropping was down in such a way that the sky in a scene from the top of the image was removed and the hood of the car from the bottom was also removed. In both instance, no gain from including those sections of the image are expected so they are effectively wasted pixels. Scaling was performed as a means of compressing the problems and increasing runtime and learning speed.

## Real-time Augmentation

The real-time augmentation step consists of various operations performed on images and labels during the training phase. This was implemented in a python generator which is used to load augmented images and labels into memory *one batch a time*. The augmentation process is performed on each batch which itself is randomly sampled from the entire training set. The operations consist of:
*  Brightness - the brightness of an image was modified by scaling it with a multiplier sampled from a uniform distribution. The idea is to make the model learn the required features for the task regardless of whether an image is dark or light.
* Horizontal flip - each image was horizontally (left/right) flipped at random by performing a coin toss. The sign on the steering angle was also flipped. The motivation is to increase the number of images to learn from by realizing that if the scene is horizontally flipped, the features to be learned should be the same.
* Steering variation - each steering label has an offset added to it that is sampled from a normal distribution of a *small* standard deviation. This effectively increases the number of training samples to learn. The justification for this is that the steering command from a driver is not a deterministic process and has some randomness in it. Therefore, it is to be expected that two hypothetically identical scenes may have distinct, albeit similar values. By adding some gaussian noise to the data we can grow the number of training samples while incorporate while 'showing' the model that there is uncertainty to be expected of the steering angle values.

Since these augmentation techniques are themselves random and changing every during every batch pass, we are in essence introducing mechanism that minimizes over-fitting. Like drop-outs which randomly shut-off nodes in a forward-pass of the model, these augmentation techniques randomly add noise and modifications to the images which prevent the model from over-fit to a specific set of images.

With these modifications the dataset is increased to 25,000 images from 10,725.
