import numpy as np
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import pickle
from tifffile import imread
import sys
from helpers.plotCalcium import *
from helpers.skeletonise import * 
from helpers.statistics import *
from helpers.load_process_time_series import *
import warnings
warnings.filterwarnings('ignore')

# code to process the behavioral data and evetaully link them to calcium dynamics data
npoints = 50 # make sure to put the same value as there is in skeletoniseAll.py
#camSize = 6.5
#binning = 2
#magnification = 20
pixToUm = ...#camSize * binning / magnification

# load the skeletons, correct the offset position, draw the head and the tail trajectory
path_save = 'path_to_save_folder'

overwrite = False
path = 'path_to_tiff_file'
fname0 = 'tiff_file_name.tif'
fname = fname0[:-4]

fname_save = os.path.join(path_save, fname)
saveFile = fname_save + '_analysed.pkl'

skip = False
if os.path.isfile(saveFile):
    if overwrite:
        os.remove(saveFile)
    else:
       #raise(FileExistsError) 
       print(saveFile + 'exists! skipping')
       skip = True
if not skip:
    try:
        image = imread(os.path.join(path, fname0), key=0)
        height = image.shape[0]
        print('path: ', os.path.join(path, 'skeletons'))
        skeleton, fps = load_skeletons(path, fname0, npoints)
        skeleton = correct_head_tail_neural_data(skeleton, os.path.join(path, fname.replace('-', '_')), height)

        # from the offsetted skeletons find the tangential and normal components of the velocity
        print('velocity components')
        velocity, velocity_units = subtract_velocity(skeleton, pixToUm, fps)
        print('velocity done...')
        bodyPart = 1
        fileName = os.path.join(path, fname0)

        # plot the activity of the each TRN and the two components of the velocity
        section = 2
        ratio, velocity, neurons = ca_activity_velocity(skeleton, velocity_units, path, fname0, fname_save, section, height, fps, pixToUm)
        
        # save ratio, velocity, neurons
        stats = {'ratio': ratio, 'velocity': velocity,
        'neurons': neurons}
        
        with open(saveFile, 'wb') as fp:
            pickle.dump(stats, fp)
        print('completed!')
    except FileNotFoundError:
        #print('skel file missing')
        pass
    except Exception as e:
        #print(e)
        print('issue with ', fname)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fnameErr = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fnameErr, ', in line: ', exc_tb.tb_lineno)
        #i = input('carry on?')


        