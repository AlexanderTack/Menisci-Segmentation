# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# This program is segmenting the menisci from 3D waterexcited dual-echo steady-state MRIs.
#
# The code is loosely based on the code developed by Marko Jocic and uploaded on github (https://github.com/jocicmarko/ultrasound-nerve-segmentation).
# Marko Jocic originally developed the code for the ultrasound segmentation challenge (https://www.kaggle.com/c/ultrasound-nerve-segmentation, MIT license).
#
# Copyright (C) 2017 Alexander Tack, Felix Ambellan
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os

import subprocess
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import re
from datetime import datetime
from multiprocessing import Pool
import pickle
import math
import kdtree
import fnmatch
from skimage import measure
from scipy import ndimage

import scipy.spatial.distance 
import helper_unet_architecture as kh

# Theano image ordering
K.set_image_dim_ordering('th')


def tryint(s):
    try:
        return int(s)
    except:
        return s

    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+).;', s) ]


def segment_patient(subDir, rel_path, patient_result_dir):
    current_mri_path = os.path.join(oai_path, rel_path)

    if not os.path.exists(patient_result_dir):
        os.makedirs(patient_result_dir)

    # Load data. Names are needed later for saving the segmentation masks
    imgs_test, img_test_names = kh.read_testing_data(current_mri_path, image_rows, image_cols)

    if(imgs_test is None):
        print('Error: Test image could not be loaded. Aborting.')
        return

    print('-' * 30)
    print('Predicting masks...')
    # create segmentation masks for each 2D Slice of imgs_tests
    model_prediction = model.predict(imgs_test, batch_size=50, verbose=1)

    # write result
    lp = 0
    for image_name in img_test_names:
        # Important! The ID should be separated by a dash. Example for one subject: 9003406-001.dcm, 9003406-002.dcm, ... , 9003406-160.dcm
        img_id = image_name
        # Important! There should be only one dot in the image name! TODO: Reverse image name, split at dot, remove first element and keep the rest
        image_pred_name = image_name + '_pred.dcm'
        referenceDicom = dicom.read_file(os.path.join(current_mri_path, image_name))

        kh.write_dicom(referenceDicom, model_prediction[lp,0,:,:], os.path.join(patient_result_dir, image_pred_name) )

        lp += 1
    return


image_rows = 384
image_cols = 384

# LEFT or RIGHT
oai_side = sys.argv[1]

# v00, v12, ... v96
oai_tp = sys.argv[2]

base_path = ""

# a file stating the OAI path relative to the OAI folder, e.g. /v00/OAI/ for v00
case_List_File = ""

# e.g. "menisci-2D-UNet.384x384.TrainedOnFirst44.hdf5"
model_path = 
print('Using model: {0}\n'.format(model_path))

# Create the 2D U-Net architecture
model = kh.create_segnet_architecture(image_rows, image_cols)
print('Model compiled.')
# Load pre-trained 2D U-Net weights
model.load_weights(model_path)
print('Loaded model weights from {0}.'.format(model_path))

# result folder
result_data_folder_path = ""
print(result_data_folder_path)

# image data: where is the OAI data/ the DICOM data
oai_path = ""
print(oai_path)


if __name__ == '__main__':

    print('Test data file: {0}'.format(case_List_File))
    
    # rows, columns = subprocess.check_output(['stty', 'size']).decode().split()
    konsoleWidth=30

    # Get all patient subdirectories
    with open(case_List_File) as f:
        subdirectories = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    subdirectories = [x.strip() for x in subdirectories]
    subdirectories.sort(key=alphanum_key)
    numSubjects = len(subdirectories)

    print('-'*(konsoleWidth-1))
    print('There are {0} testing directories.'.format(numSubjects))

    # Image Dimensions
    subvolume_rows = 64
    subvolume_cols = 64
    subvolume_z = 16

# For each patient:
# - Extract subvolumes
# - segment each subvolume
# - fuse all subvolumes employing majority voting
# - save segmentation mask as DICOM stack
    currentPatient = 0
    for subDir in subdirectories:
         print(subDir)
         patient_ID = subDir.split(" ")[0].split("/")[1]
         print("{0} of {1}".format(currentPatient, numSubjects) )

         patient_result_dir = os.path.join(result_data_folder_path, subDir.split(" ")[0], "Meniskus-2D/")
        
         segment_patient(patient_ID, subDir.split(" ")[0], patient_result_dir)
         currentPatient += 1
        
