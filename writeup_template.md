#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_nvidia.py containing a trained convolution neural network (nVidia)
* model_comma.py containing a trained convolution neural network (comma.ai)
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
# --- OR ---
python drive.py model_comma.json
# --- OR ---
python drive.py model_nvidia.json
```
I couldn't run the original file, and after referencing this [discussion](https://github.com/fchollet/keras/issues/2386) on Keras, I modified the code from
```
model = load_model('model.h5')
```

to this:
```
model.compile("adam", "mse")
weights_file = args.model.replace('json', 'h5')
model.load_weights(weights_file)
```
[//]: # (Image References)

[image1]: ./examples/2000_org.jpg "Original Image"
[image2]: ./examples/2000_read.png "Read"
[image3]: ./examples/2000_gamma.png "Gamma"


####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Training Data
I tried to collect samples myself with a keyboard, by running a few laps keeping the car at the centre as well as I can and then did some recovery actions (showing the network what to do when the car is almost off the track) for a few more laps. It didn't go so well and the car ran off track before even making it to the first turn. Then I checked out CarND's forum and some people suggested using a joystick gives better data. I tried using my XBOX 360 controller but I couldn't configure it for the simulation software in Windows/Mac/Ubuntu. I tried using my keyboard a few more times but I finally had to admit defeat and used the sample data from Udacity instead. That means I have 8036 samples. (You can download the dataset here: [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip))

### Data Augmentation
I only a little processing on the data: 
* each sample contains the `center`, `left` and `right` images. That means I actually have 24108 samples. 
* I'm only considering the `steering angle` for each image, and adding an `-0.8` offset for an `left image` and `+0.8` offset for an `right image`.
* I cropped the top area of the image which mainly shows the sky, I got this idea from the lane finding assigment we all did a few months back. This way the network can focus more on the lanes.
* I also resized the images from `160 by 320` to `32 by 64`, and since I cropped the images, it should be `20 by 64`. This will speed up the training process a lot.
* Since 24108 samples seemd to be not enough (maybe), I flipped the images to double the samples and see if this can solve the left turning problem since the track consists of left turns.
* And finally, I adjust the gamma of the images to make the constrast a bit more distinct.

Like others, I also considered doing a lot of image processing, moving the image around, flipping the images horizontally, warping. I tried all of them but none of them works.

Example:
* Original image:
* ![alt text][image1]

* Image read:
* ![alt text][image2]

* Image after resize and gamma adjustment:
* ![alt text][image3]

### Overfitting and Optimization Strategy
* I shuffled and split 10% of the data as the validation set, so 45805 samples for training, 2411 samples for validation. 
* I used an adam optimizer for the model so that manually training the learning rate wasn’t necessary.

### Model Architecture and Training Strategy
Before going over the model, I also added the following lines before building the model because I kept getting `Error: range exceeds valid bounds`, I fixed it after reading some potential solutions on another [discussion](https://github.com/fchollet/keras/issues/2681) of Keras.

Since I belong to the November cohort, there're already a bunch of proposed methods, which mainly consist of [nVidia's](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)  and [Comma.ai's] (https://github.com/commaai/research/blob/master/train_steering_model.py)

I decided to try both. Since I had cropped and resized the input images so I decided to simplify nVidia's model a bit, like this, which speeds up training and still gives me high accuracy.
Train on 45805 samples, validate on 2411 samples

#### Simplified nVidia Model

| Layer (type)                    |     Output Shape    | Param # |
|---------------------------------|---------------------|---------|
| batchnormalization_1(BatchNormalization) | (None, 20, 64, 3) | 40 | 
|convolution2d_1 (Convolution2D) | (None, 9, 31, 16) | 448 |
|convolution2d_2 (Convolution2D) | (None, 7, 15, 24) | 3480 |
|convolution2d_3 (Convolution2D) | (None, 5, 13, 36) | 7812 |
|convolution2d_4 (Convolution2D) | (None, 4, 12, 48) | 6960 |
|convolution2d_5 (Convolution2D) | (None, 3, 11, 48) | 9264 |
|maxpooling2d_1 (MaxPooling2D) | (None, 1, 5, 48) | 0 |
|flatten_1 (Flatten) | (None, 240) | 0 |
|dense_1 (Dense) | (None, 512) | 123392 |
|dropout_1 (Drouout) | (None, 512) | 0 |
|dense_2 (Dense) | (None, 100) | 51300 |
|activation_1 (Activation) | (None, 100) | 0 |
|dense_3 (Dense) | (None, 50) | 5050 |
|activation_2 (Activation) | (None, 10) | 0 |
|dense_4 (Dense) | (None, 10) | 510 |
|activation_3 (Activation) | (None, 10) | 0 |
|output (Dense) | (None, 1) | 11 |
**Total params:** 208267

#### nVidia Model Result
| Epoch                    |     Time |  loss    | val_loss |
|--------------------------|----------|----------|----------|
| 1                        |     154s | 0.0187 | 0.0149 |
| 2 | 150s | 0.0150 | 0.0136 |
| 3 | 152s | 0.0128 | 0.0110 |
| 4 | 259s | 0.0110 | 0.0101 |
| 5 | 313s | 0.0105 | 0.0100 |
| 6 | 314s | 0.0102 | 0.0096 |
| 7 | 294s | 0.0099 | 0.0094 |
| 8 | 269s | 0.0097 | 0.0092 |
| 9 | 309s | 0.0096 | 0.0091 |
| 10 | 147s | 0.0095 | 0.0091 |
| 11 | 367s | 0.0093 | 0.0089 |
| 12 | 96s | 0.0092 | 0.0088 |
| 13 | 93s | 0.0092 | 0.0088 |
| 14 | 91s | 0.0090 | 0.0087 |
| 15 | 132s | 0.0090 | 0.0087 |
| 16 | 171s | 0.0089 | 0.0085 |
| 17 | 147s | 0.0088 | 0.0086 |
| 18 | 115s | 0.0087 | 0.0085 |
| 19 | 114s | 0.0087 | 0.0085 |
| 20 | 202s | 0.0087 | 0.0085 |

#### Comma.ai Model
| Layer (type)                    |     Output Shape    | Param # |
|---------------------------------|---------------------|---------|
| batchnormalization_1(BatchNormalization) | (None, 20, 64, 3) | 40 | 
|lambda_1 (Lambda) | (None, 20, 64, 3) | 0 |
|Conv1 (Convolution2D) | (None, 5, 16, 16) | 3088 | 
|Conv2 (Convolution2D) | (None, 3, 8, 32) | 12832 | 
|Conv3 (Convolution2D) | (None, 2, 4, 64) | 51264 |
|flatten_1 (Flatten) | (None, 512) | 0 |
|dropout_1 (Dropout) | (None, 512) | 0 |
|elu_1 (ELU) | (None, 512) | 0 | 
|FC1 (Dense) | (None, 512) | 262656 |
|dropout_2 (Dropout) | (None, 512) | 0 |
|elu_2 (ELU) | (None, 512) | 0 |
|output (Dense) | (None, 1) | 513 | 
**Total params:** 330353

#### Comma.ai Result
| Epoch                    |     Time |  loss   | val_loss |
|--------------------------|----------|---------|----------|
| 1 |154s|0.0187|0.0149|
| 2 |150s|0.0150|0.0136|
| 3 |152s|0.0128|0.0110|
| 4 |259s|0.0110|0.0101|
| 5 |313s|0.0105|0.0100|
| 6 |314s|0.0102|0.0096|
| 7 |294s|0.0099|0.0094|
| 8 |269s|0.0097|0.0092|
| 9 |309s|0.0096|0.0091|
| 10 |1473s|0.0095|0.0091|
| 11 |367s|0.0093|0.0089|
| 12 |96s|0.0092|0.0088|
| 13 |93s|0.0092|0.0088|
| 14 |91s|0.0090|0.0087|
| 15 |132s|0.0090|0.0087|
| 16 |171s|0.0089|0.0085|
| 17 |147s|0.0088|0.0086|
| 18 |115s|0.0087|0.0085|
| 19 |114s|0.0087|0.0085|
| 20 |202s|0.0087|0.0085|

For both models, I ran 20 epochs and added early stopping mechanism.

### Final Adjustment
I found out training the model merely is not enough, I spent hours tweaking the values of `steering_angle` and `throttle` to make sure the car is not steering too fast or running too quickly.

## Result:
### Comma.ai
I suggest you watching it in 2x speed, the car moves really slow.  
To see the part where the car is actually moving, please skip to 0:28.  
[![IMAGE ALT TEXT](http://img.youtube.com/vi/DZmIwV8ADGw/0.jpg)](https://www.youtube.com/watch?v=DZmIwV8ADGw "Self-driving Car: Behavior Cloning (Comma.AI)")
### nVidia
I suggest you watching it in 2x speed, the car moves really slow.  
To see the part where the car is actually moving, please skip to 1:08.  
[![IMAGE ALT TEXT](http://img.youtube.com/vi/S9x58PpZP7M/0.jpg)](https://www.youtube.com/watch?v=S9x58PpZP7M&t=6s "Self-driving Car: Behavior Cloning (nVidia)")



