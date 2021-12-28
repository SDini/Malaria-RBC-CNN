##################################################
## In this project, I create a CNN for classifying
## RBC images into malaria infected and uninfected
##################################################
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shelve
import sklearn
import tensorflow.keras as tf
from matplotlib.image import imread

## Set the working directory
try:
    os.chdir("/Users/sdini/PycharmProjects/TensorFlow")
except:
    os.chdir("/mnt/TensorFlow")

## Load the images
images_dir = "cell_images"

os.listdir(images_dir)

train_dir = images_dir+"/train/"
test_dir = images_dir+"/test/"

image_tmp = plt.imread(images_dir+"/train/parasitized/C176P137NThinF_IMG_20151201_114235_cell_13.png")

try:
    plt.imshow(image_tmp)
except:
    print("no graphic device available")

print(image_tmp.shape)

### Importing keras and doing the analysis
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Activation, Dense, Dropout

# datagen = ImageDataGenerator(featurewise_center=True,
#                              featurewise_std_normalization=True,rotation_range=20,
#                              width_shift_range=0.2,height_shift_range=0.2,
#                              horizontal_flip=True,validation_split=0.2)

datagen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               # rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

datagen.flow_from_directory(train_dir)
datagen.flow_from_directory(test_dir)

model_1 = Sequential()

model_1.add(Conv2D(filters = 32,
                   kernel_size = (3,3),
                   #strides = (2,2),
                   #padding = 'valid',
                   activation = 'relu',
                   input_shape=(130, 130, 3)),
            )
model_1.add(MaxPool2D((2,2)))

model_1.add(Conv2D(filters = 64,
                   kernel_size= (3,3),
                   #strides = (2,2),
                   #padding = 'valid',
                   activation = 'relu',
                   input_shape=(130, 130, 3)),
            )
model_1.add(MaxPool2D((2,2)))

model_1.add(Conv2D(filters = 32,
                   kernel_size= (3,3),
                   #strides = (2,2),
                   #padding = 'valid',
                   activation = 'relu',
                   input_shape=(130, 130, 3)),
            )
model_1.add(MaxPool2D((2,2)))

model_1.add(Flatten())

model_1.add(Dense(128))
model_1.add(Activation('relu'))

model_1.add(Dropout(0.5))
model_1.add(Dense(1))
model_1.add(Activation('sigmoid'))

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stp = EarlyStopping(monitor='val_loss', patience=2)

batch_size = 16
train_image_flow = datagen.flow_from_directory(train_dir,
                                               target_size = (130, 130),
                                               class_mode='binary',
                                               color_mode='rgb',
                                               batch_size=batch_size)

test_image_flow = datagen.flow_from_directory(test_dir,
                                              target_size = (130, 130),
                                              class_mode='binary',
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              shuffle=False)

if True:
    out_1 = model_1.fit_generator(train_image_flow, epochs=20, callbacks=[early_stp], validation_data=test_image_flow)
    model_1.save('malariaRBC_binaryclassification.h5', )
    # Saving the objects:
    filename =   os.getcwd()+'/shelve.out'
    my_shelf = shelve.open(filename, 'n')  # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()
else:
    from tensorflow.keras.models import load_model
    model_1 = load_model("malariaRBC_binaryclassification.h5")

if False:
    model_1.evaluate_generator(test_image_flow)
    from sklearn.metrics import classification_report,confusion_matrix

