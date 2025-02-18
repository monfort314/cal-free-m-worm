import sys
sys.path.append("..")
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import xml.etree.ElementTree as et
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import signal
import cv2
from tifffile import imread, imwrite
import scipy.stats as st
from IPython.display import display, clear_output
import time

def smooth_spline_nans(x, y, s):    
    N = len(x)
    t = np.linspace(0, 1, N)
    idxValid = ~np.isnan(x + y)
    tck, u = interpolate.splprep([t[idxValid], x[idxValid], y[idxValid]], s=s)
    out = interpolate.splev(u, tck)
    Xout = np.nan * np.zeros(N)
    Xout[idxValid] = out[1]   
    Yout = np.nan * np.zeros(N)
    Yout[idxValid] = out[2]
    return Xout, Yout
 
def smooth_spline(x, y):
    N = len(x)
    t = np.linspace(0, 1, N)
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out[0], out[1]
    
def moving_average(a, n=5):
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan
    return ret

def moving_average_nans(x, n=5):
    mx = np.ma.masked_array(x,np.isnan(x))
    y = moving_average(mx, n)
    return y

def fill_in_nans(trackData, N=None):
    if N == None:
        nframes = int(np.max(trackData[:, 0])) + 1
    else:
        nframes = N
    track = np.nan * np.zeros(shape = (nframes, 2))
    for i in range(trackData.shape[0]):
        try:
            it = int(trackData[i, 0])
            track[it] = trackData[i, 1:]
        except:
            pass
    return track

def read_spots(spotsRoot):
    AllSpots = {}
    nframes = len(spotsRoot)
    for i in range(nframes):
        nspotsInFrame = len(spotsRoot[i])
        for j in range(nspotsInFrame):
            ID = spotsRoot[i][j].attrib['ID']
            t = int(spotsRoot[i].attrib['frame'])
            x = float(spotsRoot[i][j].attrib['POSITION_X'])
            y = float(spotsRoot[i][j].attrib['POSITION_Y'])
            AllSpots[ID] = [t, x, y]
    return AllSpots

def load_xml_trajs(xmlfile):
    """ 
    Load xml files into a python dictionary with the following structure:
        tracks = {'0': {'nSpots': 20, 'trackData': numpy.array(t, x, y) }}
    Tracks should be xml file from 'Export tracks to XML file',
    that contains only track info but not the features.
    Similar to what 'importTrackMateTracks.m' needs.
    """
    try:
        tree = et.parse(xmlfile);
    except OSError:
        print('Failed to read XML file {}.'.format(xlmfile) )
    root =  tree.getroot()
    spotsRoot = root[1][1]
    AllSpots = read_spots(spotsRoot)
    trackRoot = root[1][2]
    # print(root.attrib)  # or extract metadata
    nTracks = len(trackRoot)
    tracks = {}
    for i in range(nTracks):
        trackIdx = str(i)
        tracks[trackIdx] = {}
        nSpots = int(trackRoot[i].attrib['NUMBER_SPOTS'])
        tracks[trackIdx]['nSpots'] = nSpots
        trackData = np.array([ ]).reshape(0, 3)
        for j in range(nSpots-1):
            source = trackRoot[i][j].attrib['SPOT_SOURCE_ID']
            target = trackRoot[i][j].attrib['SPOT_TARGET_ID']
            trackData = np.vstack((trackData, AllSpots[source]))
            trackData = np.vstack((trackData, AllSpots[target]))
        tracks[trackIdx]['trackData'] = trackData
    return tracks


def get_spots(xmlFile):
    tree = et.parse(xmlFile)
    root = tree.getroot()
    allSpots = root[1][1]
    nFrames = len(allSpots)
    Spots = {}
    for i in range(nFrames):
        for ispot in range(len(allSpots[i])):
            ID = allSpots[i][ispot].attrib['ID']
            nth = int(allSpots[i][ispot].attrib['FRAME'])
            x = float(allSpots[i][ispot].attrib['POSITION_X'])
            y = float(allSpots[i][ispot].attrib['POSITION_Y'])
            Spots[ID] = [nth, x, y]
    return Spots

def get_tracks(xmlFile):
    Spots = getSpots(xmlFile)
    tree = et.parse(xmlFile)
    root = tree.getroot()
    allTracks = root[1][2]
    nTracks = len(allTracks)
    N = int(root[2][0].attrib['nframes'])
    Tracks = np.nan * np.zeros(shape=(nTracks, N, 2))
    for i in range(nTracks):
        nspots = len(allTracks[i])
        for ispot in range(nspots):
            source = allTracks[i][ispot].attrib['SPOT_SOURCE_ID']
            target = allTracks[i][ispot].attrib['SPOT_TARGET_ID']
            (nth, x, y) = Spots[source]
            Tracks[i, nth] = (x, y)
            (nth, x, y) = Spots[target]
            Tracks[i, nth] = (x, y)
    return Tracks


def crop_roi(red, green, radius, x, y):
    mask1 = np.zeros(shape=red.shape, dtype=np.uint8)
    mask1 = cv2.circle(mask1, (x, y), radius, (255, 255, 255), -1)
    roiRed = np.zeros_like(red)
    roiRed[mask1 == 255] = red[mask1 == 255]
    roiRed = roiRed[y - radius: y + radius, x - radius : x + radius]
    roiGreen = np.zeros_like(green)
    roiGreen[mask1 == 255] = green[mask1 == 255]
    roiGreen = roiGreen[y - radius: y + radius, x - radius : x + radius]
    return roiRed, roiGreen

def match_stage_time_stamp(t, x, y, tC, N):
    # t, x, y stage timestamp (sec)/position
    # tC: camera timestamp (in sec)

    txC = np.array(range(len(tC)))
    t = t - tC[0]
    tC = tC - tC[0]

    x = x - x[0]
    y = y - y[0]

    # first upsample
    spl = interp1d(txC, tC, kind='linear', fill_value=(tC[0], tC[-1]))
    txC2 = np.linspace(txC[0], txC[-1], 2 * N)
    tC2 = spl(txC2)

    # interpolate and dowsample x
    intx = interp1d(t, x, kind='linear', bounds_error=False, fill_value=(x[0], x[-1]))
    tmp = intx(tC2)
    xnew = tmp[::2]#signal.resample(tmp, N)

    # interpolate and dowsample y
    inty = interp1d(t, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
    tmp = inty(tC2)
    ynew = tmp[::2] #signal.resample(tmp, N)
    return xnew, ynew