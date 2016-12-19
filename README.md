# Network Architecture Overview

||Image Preprocessing|# of Conv Layers|# of FC Layers|MaxPooling|Dropout|
|---|---|---|---|
|Alpha1|RGB|3|3|None|None|
|Alpha2|YUV|3|3|None|None|

# Notes
* To remove a bias towards driving straight the training data includes a higher proportion of frames that represent road curves... i.e. balance training data on curvature
* Flip images?
* elu? A type of activation function?
* Drop training set images if steering is too low?
* Generators? python/keras?
* 1x1 filters on raw images to learn a color map
* YUV colorspace
* Sample video at 10 FPS: A higher sampling rate would result in including images that are highly similar and thus not provide much useful information.
