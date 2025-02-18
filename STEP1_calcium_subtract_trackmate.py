# read and process videos from trackmate

import sys
sys.path.append("..")
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import xml.etree.ElementTree as et
import pandas as pd
from scipy import interpolate
import cv2
from tifffile import imread, imwrite
import scipy.stats as st
from helpers.load_process_time_series import *
import zarr


# calibration data/constants
overwrite = False
#camSize = 6.5
#binning = 2
#magnification = 20
pixToUm = ... #camSize * binning / magnification
radius = 10
fps = ... #8

image_path = 'path_to_tiff_image.tif'
labelData = pd.read_csv(r'NeuronLabels.csv')
xml = 'path_to_trackmate_file.xml'

base = os.path.split(xml)[-1]
base = base[:-4]

print('reading file')
image = imread(image_path)
N, nchann, height, width = image.shape
print('shape', image.shape)
Red = image[:, 1, :, :]
Green = image[:, 0, :, :]

t = np.arange(N) / fps

# load track data
tracks = load_xml_trajs(xml)

# load stage data
stage_data = 'stage_file_data.txt'
data = pd.read_csv(stage_data, sep='\t', header=None)
x = np.array(data[3])
y = np.array(data[2])
tS = np.array(data[1])

# load the camera timestep 
camera_timestamp_data_file = 'fname_cam.txt'
dataCam = pd.read_csv(camera_timestamp_data_file, sep='\t', header=None)
tC = np.array(dataCam[1])

x, y = match_stage_time_stamp(tS, x, y, tC, N)
dx = np.append(np.diff(x),0)
dy = np.append(np.diff(y),0)

xpix = x / pixToUm
ypix = y / pixToUm
xpix = xpix - np.min(xpix)
ypix = ypix - np.min(ypix)

# pick the neuron
path0 = os.path.split(image_path)[0]

strFile = os.path.join(path0, base + '.xlsx')
strFile = strFile.replace('-', '_')
with pd.ExcelWriter(strFile, engine="openpyxl") as writer:
    idxTable = labelData.index[(labelData['path'] == path0) & (labelData['tracks file'] == os.path.split(xml)[-1])].tolist()
    print(idxTable)
    for j in idxTable:
        neuronName = labelData['TRN'][j]
        ID = labelData['track ID'][j]
        print('neuron ' + neuronName + ' ' + str(ID))
        traj = fill_in_nans(tracks[str(ID)]['trackData'], N)
        trajUm = traj * pixToUm

        signalRed = np.nan * np.zeros(N)
        signalGreen = np.nan * np.zeros(N)
        ratio = np.nan * np.zeros(N)
        
        for i in range(N):    
            red = Red[i]
            green = Green[i]
            
            roiRed = bckgRed = wormRed = np.nan
            roiGreen = bckgGreen = wormGreen = np.nan

            tx = traj[i, 0]
            ty = traj[i, 1]
            if ~np.isnan(tx):
                tx = int(tx)
                ty = int(ty)
                try:
                    roiRed, roiGreen = crop_roi(red, green, radius, tx, ty)
                    bckgRed = np.nanmean(red[red < st.scoreatpercentile(red, 20)])
                    bckgGreen = np.nanmean(green[green < st.scoreatpercentile(green, 20)])

                    wormRed = np.nanmean(red[(red <= st.scoreatpercentile(red, 70)) & (red >= st.scoreatpercentile(red, 50))])
                    wormGreen = np.nanmean(green[(green <= st.scoreatpercentile(green, 70)) & (green >= st.scoreatpercentile(green, 50))])
                except: 
                    pass
            
            if ~np.isnan(roiRed).all():
                signalRed[i] = (np.nanmean(roiRed) - bckgRed) / (wormRed - bckgRed)
                signalGreen[i] = (np.nanmean(roiGreen) - bckgGreen) / (wormGreen - bckgGreen)
                ratio[i] = signalGreen[i] / signalRed[i]
            
        locx = traj[:, 0] + xpix
        locy = height - traj[:, 1] + ypix
        data = {}
        #data['timestamp'] = tC - tC[0]
        data['x'] = locx
        data['y'] = locy
        data['red'] = signalRed
        data['green'] = signalGreen
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=neuronName)

print('done')
