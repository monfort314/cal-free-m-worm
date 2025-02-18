
import sys
sys.path.append("..")
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import interpolate

def smooth_spline(x, y, N = None):
    if N == None:
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

def fill_in_nans(track_dData, N=None):
    if N == None:
        nframes = int(np.max(track_data[:, 0])) + 1
    else:
        nframes = N
    track = np.nan * np.zeros(shape = (nframes, 2))
    for i in range(track_data.shape[0]):
        try:
            it = int(track_data[i, 0])
            track[it] = track_data[i, 1:]
        except:
            pass
    return track
