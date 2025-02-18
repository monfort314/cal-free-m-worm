# Now look at the skeletons


import os
from tifffile import imread, imwrite
import scipy.stats as st
import cv2
import sys
import numpy as np
#print(sys.version)
import imutils
import matplotlib.pyplot as plt
import glob 
import zarr

from helpers.load_preprocess import * 
from helpers.separate_worm_bckg import *
from helpers.skeletonise import *  

binning_additional = 2
denoising_strength = 10
edge_threshold = (2, 10)
iterations_dilation = 1
npoints = 50
desired_length = 1000 // binning_additional
deviation = 300 // binning_additional
from_threshold = True        
percmin = 0.001
percmax = 99.99
threshold = 11

overwrite = False
show = True
tryToCorrect = False
# values for adjusting the contrast in the red channel
valMax = 4000 
valMin = 0
path = 'path_to_tiff_file'
path_save = os.path.join(path, 'skeletons')
if not os.path.isdir(path_save):
    os.mkdir(path_save)

fname0 = 'file_name.tif'
fname0 = os.path.split(fname)[-1][:-4]
fnameCSV = os.path.join(path_save, fname0 + '_skel.csv')
exists = os.path.isfile(fnameCSV)
if not exists or overwrite: 
    fname = os.path.join(path, fname0)
    print('Loading ' + fname)
    try:     
        lazytiff = imread(fname, aszarr = True)
        image0 = zarr.open(lazytiff, mode='r') 
        N, nchann, height, width = image0.shape   
        print('done!')
        im = image0[0]
        tagRFP =  image0[:, 1, :, :]
        lazytiff.close()
        image0 = resize(tagRFP, binning_additional)

        print('Adjusting contrast...')
        image = adjust_contrast(image0, valMin, valMax, fromPerc=False)
        print('done!')

        frame = image[1].copy()
        threshold = np.ones(N) * threshold

        ROI = identify_worm(image, from_threshold=from_threshold, threshold=threshold, denoising_strength=denoising_strength, 
            edge_threshold=edge_threshold, iterations_dilation=iterations_dilation, show=show)
        print('done!')

        print('Skeletonising...')
        
        SkelArray, S, IsSkelGood, Frame = skeletonise_array(ROI, npoints, desired_length=desired_length, deviation=deviation, image=image)
        
        idx = ~IsSkelGood
        
        SkelArray = SkelArray * binning_additional
        skel_reshaped = SkelArray.reshape(SkelArray.shape[0], -1)
        if os.path.isfile(fnameCSV):
            os.remove(fnameCSV)
        np.savetxt(fnameCSV, skel_reshaped)
        print('done')
    except KeyboardInterrupt:
        print('Interrupted')
        cv2.destroyAllWindows()
        raise(KeyboardInterrupt)
    except Exception as error:
        print('issue with ' + fname)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, ', in line: ', exc_tb.tb_lineno)
