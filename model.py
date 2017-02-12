# Import necessary libraries
import numpy as np
import pandas as pd
import os
import json
from skimage.exposure import adjust_gamma
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Lambda, ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.misc import imresize

from keras import backend as K

K.set_image_dim_ordering('tf')

def create_nvidia_model():
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()

    return model

def create_comma_ai_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(20, 64, 3)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='relu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='relu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='relu', name='Conv3'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='relu', name='FC1'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name='output'))
    model.summary()

    return model


def read_data(mode, file_name):
    data_dir = os.getcwd() + '/'
    print('# -------- Reading ' + mode + ' Data ---------- #')
    columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
    input_data = pd.read_csv(file_name, names=columns)

    num_of_data = len(input_data)

    angles = np.array(input_data['steering_angle'])

    X_train = np.ndarray(shape=(len(angles)*3, 20, 64, 3))
    y_train = []

    input_data = np.copy(input_data)

    count = 0;
    for line_data in input_data[:]:
        # There's an extra line which stores the keys ('center', 'left', 'right', 
        # 'steering_angle', 'throttle', 'brake', 'speed')
        # here I am removing this particular line
        if line_data[0] != "center":
            if (count % 200 == 0):
                print(count)
            
            if (mode == 'Udacity'):
                center_image = ndimage.imread(data_dir + line_data[0].strip()).astype(np.float32)
                left_image = ndimage.imread(data_dir + line_data[1].strip()).astype(np.float32)
                right_image = ndimage.imread(data_dir + line_data[2].strip()).astype(np.float32)
            else:
                center_image = ndimage.imread(line_data[0].strip()).astype(np.float32)
                left_image = ndimage.imread(line_data[1].strip()).astype(np.float32)
                right_image = ndimage.imread(line_data[2].strip()).astype(np.float32)
                                    
            X_train[count] = imresize(center_image, (32,64,3))[12:,:,:]
            y_train.append(float(line_data[3]))
            count = count + 1

            X_train[count] = imresize(left_image, (32,64,3))[12:,:,:]        
            y_train.append(float(line_data[3]) - .08)
            count = count + 1

            X_train[count] = imresize(right_image, (32,64,3))[12:,:,:]        
            y_train.append(float(line_data[3]) + .08)
            count = count + 1

    y_train = np.array(y_train)
    
    # a line is removed from input_data, 
    # so X_train should have three lines of empty data
    # here I'm cleaning it up
    X_train = X_train[:y_train.shape[0]]

    # Adjust gamma in images to increase contrast
    X_train = adjust_gamma(X_train)

    return X_train, y_train

# Create a mirror image of the images in the dataset to combat left turn bias
def create_mirror_data(X_train, y_train):
    mirror = np.ndarray(shape=(X_train.shape))
    count = 0
    for i in range(len(X_train)):
        mirror[count] = np.fliplr(X_train[i])
        count += 1
    mirror.shape

    # Create mirror image labels
    mirror_angles = y_train * -1

    return mirror, mirror_angles



def read_sample_data_only():
    X_train, y_train = read_data('Udacity', 'udacity_driving_log.csv')
    mirror, mirror_angles = create_mirror_data(X_train, y_train)
    X_train = np.concatenate((X_train, mirror), axis=0)
    y_train = np.concatenate((y_train, mirror_angles),axis=0)

    return X_train, y_train


def read_custom_data_only():
    X_train, y_train = read_data('Custom', 'driving_log.csv')
    mirror, mirror_angles = create_mirror_data(X_train, y_train)
    X_train = np.concatenate((X_train, mirror), axis=0)
    y_train = np.concatenate((y_train, mirror_angles),axis=0)

    return X_train, y_train

def read_all_data():
    X_train, y_train = read_sample_data_only()
    X_train_c, y_train_c = read_custom_data_only()

    X_train = np.concatenate((X_train, X_train_c), axis=0)
    y_train = np.concatenate((y_train, y_train_c),axis=0)

    return X_train, y_train


# Read the data

print('# ------------------------- Start Reading Data ------------------------- #')
print(' Images will be resized to 32x64 to increase training speeds')
print(' The top 12 pixels are cropped off ' ) # because they contain no useful information for driving behavior
print(' Gamma of each image is adjusted to increase contrast')
X_train, y_train = read_sample_data_only()

print('# --------------- Start Generating Training/Testing Test --------------- #')
print(' Data is split, with 10 percent of the data as the test set')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.1)

print('# ---------------------- Start Creating the Model ---------------------- #') 
# model = create_comma_ai_model()
model = create_nvidiamodel()

print('# - Start Compiling Model with Adam Optimizer and Learning Rate of .0001 - #')
# learning rate is configurable
adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam)
print(' with checkpoint to save weights whenever validation loss improves ')
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')


print(' and callback to stop training when validaition loss is non-decreasing')
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

print('# ---------- Start Training Model (20 epochs, batch size 128) ---------- #') 
model.fit(X_train,
        y_train,
        nb_epoch=20,
        verbose=1,
        batch_size=128,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, callback])

print('# ------------------------- Saving Model ------------------------- #')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save("model.h5")
