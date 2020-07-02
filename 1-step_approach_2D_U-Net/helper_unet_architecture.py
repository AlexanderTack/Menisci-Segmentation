from __future__ import print_function

import os
import cv2
import numpy as np

from random import shuffle
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import Dropout
from keras import backend as K
from keras.losses import mean_absolute_error
from keras.optimizers import Adam

import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset

# Initialize Keras-related global settings
def init_keras():
    # Theano dimension ordering in this code
    K.set_image_dim_ordering('th')

# Read in training data
# Input:
# training_data_folder_path: Path with training data. Every subject in an own subdirectory
# training_mask_data_folder_path: Path with training data masks. Every subject in an own subdirectory
def read_training_data(training_data_folder_path, training_mask_data_folder_path, image_rows, image_cols, startImage, endImage):

    training_data_set_names = os.listdir(training_data_folder_path)
    training_data_set_names.sort()
    training_data_set_names = training_data_set_names[startImage:endImage]

    training_data_mask_set_names = os.listdir(training_mask_data_folder_path)
    training_data_mask_set_names.sort()
    training_data_mask_set_names = training_data_mask_set_names[startImage:endImage]

    num_image_sets = len(training_data_set_names)
    num_mask_sets = len(training_data_mask_set_names)

    print('-' * 30)
    print('Start Subject: {0}'.format(startImage))
    print('End Subject: {0}'.format(endImage))
    print('Path MRIs: {0}'.format(training_data_folder_path))
    print('Path masks: {0}'.format(training_mask_data_folder_path))
    print('Number of subjects (MRI): {0}'.format(num_image_sets))
    print('Number of subjects (masks): {0}'.format(num_mask_sets))


    num_training_masks = 0

    # loop over all mask directories and get the total number of masks
    for directory_name in training_data_mask_set_names:
        current_mask_path = os.path.join(training_mask_data_folder_path, directory_name)
        current_mask_path_masks = os.listdir(current_mask_path)
        num_current_mask_path = len(current_mask_path_masks)
        num_training_masks += num_current_mask_path

    print('Total number of training masks: {0}'.format(num_training_masks))
    print('-' * 30)

    imgs_train = np.ndarray((num_training_masks, 1, image_rows, image_cols), dtype=np.float32)
    imgs_train_mask = np.ndarray((num_training_masks, 1, image_rows, image_cols), dtype=np.float32)

    print('Reading data...\n')

    i = 0

    for directory_name in training_data_set_names:
        current_mri_path = os.path.join(training_data_folder_path, directory_name)
        current_mris = os.listdir(current_mri_path)
        current_mris.sort()

        for mri_name in current_mris:
	    mask_id = mri_name.split('-')[0]
	    mask_number = mri_name.split('-')[2].split('.')[0]
            mask_name = mask_id + '-' + mask_number.lstrip('0')  + '.png'
            current_mask_path = os.path.join(training_mask_data_folder_path, directory_name)
            mask_path_name = os.path.join(current_mask_path, mask_name)

            if os.path.isfile(mask_path_name):
                # Load single uint16 channel image from path.
	        img = dicom.read_file(os.path.join(current_mri_path, mri_name))
	        # print(img.pixel_array.dtype)

	        if img is None or img.pixel_array.dtype != 'uint16':
                    print("ERROR READING {0}. ABORTING".format(os.path.join(current_mri_path, mri_name)))
                    return None, None

		img = img.pixel_array
		img = img.astype('float32')
		img = cv2.resize(img, (image_rows, image_cols))
		img = np.reshape(img, (1, image_rows, image_cols))

	        mask = cv2.imread(mask_path_name, cv2.IMREAD_UNCHANGED)

	        if mask is None:
                    print("ERROR READING {0}. ABORTING".format(mask_path_name))
                    return None, None

		mask = mask.astype('float32')
		mask = cv2.resize(mask, (image_rows, image_cols))
		mask = np.reshape(mask, (1, image_rows, image_cols))
		mask = mask / 255.0

                imgs_train[i] = img
                imgs_train_mask[i] = mask

                if i % 1000 == 0:
                    print('Done: {0}/{1} sets'.format(i, num_training_masks))

                i += 1


    print('Loading done for {0}/{1} training sets.'.format(i, num_training_masks))
    return imgs_train, imgs_train_mask

smoothDSC = np.finfo(float).eps

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smoothDSC) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothDSC)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Creates the UNet segmentation model (not compiled)
def create_segnet_architecture(image_rows, image_cols):
    inputs = Input((1, image_rows, image_cols))
    conv1 = Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
    D1=Dropout(0.1)(conv1)
    conv1 = Conv2D(32, (5, 5), padding="same", activation="relu")(D1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (5, 5), padding="same", activation="relu")(pool1)
    D2=Dropout(0.1)(conv2)
    conv2 = Conv2D(64, (5, 5), padding="same", activation="relu")(D2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (5, 5), padding="same", activation="relu")(pool2)
    D3=Dropout(0.1)(conv3)
    conv3 = Conv2D(128, (5, 5), padding="same", activation="relu")(D3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv5 = Conv2D(256, (5, 5), padding="same", activation="relu")(pool3)
    D5=Dropout(0.1)(conv5)
    conv5 = Conv2D(256, (5, 5), padding="same", activation="relu")(D5)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv3], axis=1)
    conv7 = Conv2D(128, (5, 5), padding="same", activation="relu")(up7)
    D7=Dropout(0.1)(conv7)
    conv7 = Conv2D(128, (5, 5), padding="same", activation="relu")(D7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (5, 5), padding="same", activation="relu")(up8)
    D8=Dropout(0.1)(conv8)
    conv8 = Conv2D(64, (5, 5), padding="same", activation="relu")(D8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (5, 5), padding="same", activation="relu")(up9)
    D9=Dropout(0.1)(conv9)
    conv9 = Conv2D(32, (5, 5), padding="same", activation="relu")(D9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# Read in testing data
# path: path with image data (every subject in an own subdirectory)
def read_testing_data(path, image_rows, image_cols):

    testing_data_set_names = os.listdir(path)
    testing_data_set_names.sort()

    num_testing_images = len(testing_data_set_names)

    #print('Total number of s: {0}'.format(num_testing_images))

    imgs_test = np.ndarray((num_testing_images, 1, image_rows, image_cols), dtype=np.float32)

    #print('Reading the data...')

    i = 0
    imgs_names = []

    for mri_name in testing_data_set_names:
	# Load single uint16 channel DICOM image from path.
	img = dicom.read_file(os.path.join(path, mri_name))

	if img is None or img.pixel_array.dtype != 'uint16':
	    print("ERROR READING {0}. ABORTING".format(os.path.join(path, mri_name)))
	    return None, None

	img = img.pixel_array
	img = img.astype('float32')
	img = cv2.resize(img, (image_rows, image_cols))
	img = np.reshape(img, (1, image_rows, image_cols))

	imgs_test[i] = img
	imgs_names.append(mri_name)

	#if i % 1000 == 0:
	#    print('Done: {0}/{1} sets'.format(i, num_testing_images))

	i += 1

    print('-' * 30)
    print('Loaded {0}/{1} slices.'.format(i, num_testing_images))


    return imgs_test, imgs_names


def write_dicom(dcmRef, dcm2D, filename):
    """
    INPUTS:
    dcmRef: reference dicom image
    dcm2D: 2D numpy ndarray.
    filename: output path and name for file.
    """


    # set data type of the pixel array to uint16
    pixel_array = dcm2D[:,:]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)

    dcmRef.PixelData = pixel_array.tostring()

    dcmRef.save_as(filename)

    return






































