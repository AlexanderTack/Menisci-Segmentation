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

#os.environ['THEANORC'] = os.path.join(os.environ['HOME'], ".theanorc_local")

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

# import Z_mergeBoxes3d_helper as merger
# import Z_data_helper as datHlp
# import Z_helper_new as hlpNew

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

def segment_patient(subDir, currentPatient):
    start_time = datetime.now()
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    print('------- Patient {0}, ID: {1} -------\n'.format(currentPatient, subDir))
    current_mri_path = os.path.join(test_data_path, subDir)

# ------------- Load DICOM and mask data for the current subject
    # Load all DICOM slices and combine to 3D volume
    # print('~~~~ Step 1/5: Loading data')
    start_time_loading_data = datetime.now()
    current_mri_path_list = os.listdir(current_mri_path)
    print(current_mri_path)
    current_mri_path_real = current_mri_path + "/" + current_mri_path_list[0]
    print('\n\nID: {0}'.format(subDir))
    print(current_mri_path_real)
    print('model: {0}'.format(model_path))
    currentResultDir = meniscus_result_path_base + tp + "/" + subDir + "/"
    print('ResultDir: {0}\n'.format(currentResultDir))
    if not os.path.exists(currentResultDir):
	os.makedirs(currentResultDir)
	
    script = "python /localFour/people/bzftacka/PrevOP/scripts_meniscus/01_Step_I_2D_U-Net.py " + subDir + " " + current_mri_path_real + " " + model_path + " " + currentResultDir
    os.system(script)
    
    
    return

if __name__ == '__main__':

    rows, columns = subprocess.check_output(['stty', 'size']).decode().split()
    konsoleWidth=int(columns)

    base_path = "/vis/scratchN/bzftacka/PrevOP/"

    result_path = os.path.join(base_path, 'Segmentations/06_FC/')
    print('Result path: {0}'.format(result_path))

    model_path = "/localFour/people/bzftacka/PrevOP/scripts_meniscus/models/menisci-2D-UNet.384x384.TrainedOnFirst44.hdf5"
    print('Using model: {0}\n'.format(model_path))
    
    meniscus_result_path_base = "/vis/scratchN/bzftacka/PrevOP/Segmentations/08_2D_Meniskus/"

    # Image Dimensions
    subvolume_rows = 64
    subvolume_cols = 64
    subvolume_z = 16

# Compile the U-Net and load weights
    print('-'*(konsoleWidth-1))
    print('Creating and compiling 3D U-Net...')
    print('-'*(konsoleWidth-1))
    # model = get_unet()
    # model.load_weights(model_path)
    # print('\nLoaded model weights: {0}'.format(model_path))

    tps = os.listdir(result_path)
    for tp in tps:
        print(tp)
        test_data_path = os.path.join("/vis/scratchN/bzftacka/PrevOP/DESS/", tp)

        currentPatient = 1

        curr_tp_subdirs = os.listdir(test_data_path)
        for subDir in curr_tp_subdirs:
	    # print(subDir)
	    segment_patient(subDir, currentPatient)
	    currentPatient += 1






















