import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.stats as st
from scipy.interpolate import splev, splrep
from helpers.load_process_time_series import *
from helpers.skeletonise import *


NANTHRESH = 0.25
wsec = 5
window = wsec * fps
step = int(0.3 * window)
window = int(window)


def calculate_length(skeleton):
    N, npts, _ = skeleton.shape
    length = np.nan * np.zeros(N)
    for i in range(N):
        x = skeleton[i, :, 0]
        y = skeleton[i, :, 1]
        length[i] = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    return length

def mark_reliable_frames(skeleton_length, ratio, val_min=-1, val_max=-1, blank=15, fps=8):
    nan_idxs = np.isnan(skeleton_length)
    skeleton_length = moving_average_nans(skeleton_length, 5)
    if val_min == -1:
        val_min = st.scoreatpercentile(skeleton_length[~nan_idxs], 50)
    if val_max == -1:
        val_max = st.scoreatpercentile(skeleton_length[~nan_idxs], 90)
    ratio_smooth = moving_average_nans(ratio, n=20)
    ratio80 = st.scoreatpercentile(ratio_smooth, 80)
    idxs = (skeleton_length > val_min) & (skeleton_length < val_max)
    print(val_min, val_max)
    last_bad = -2
    for i in range(len(idxs)):
        if not idxs[i]:
            last_bad = i
        elif last_bad + 1 == i:
            if (ratio_smooth[i] > ratio80) & (ratio_smooth[i] < ratio_smooth[i - 1]):
                last_bad = min([len(idxs), i + blank * fps])
                idxs[i : last_bad] = False
                print('erasing between ', i, 'and', last_bad)
    return idxs


def corr_coeff_nans(s1, s2):
    validIdxsX = ~np.isnan(s1)
    s1 = fill_in_nans(s1)
    s2 = fill_in_nans(s2)
    return np.corrcoef(s1, s2)

def cov_nans(s1, s2):
    validIdxsX = ~np.isnan(s1)
    s1 = fill_in_nans(s1)
    s2 = fill_in_nans(s2)
    return np.cov(s1, s2)

def windowed_corr_coeff(x, y, window, step, process=False):
    n = len(x)
    CC = []
    Idxs = []
    nwins = (n - window) // step
    for i in range(nwins):
        xi = x[i * step : i * step + window]
        yi = y[i * step : i * step + window]
        if np.mean(np.isnan(xi)) < NANTHRESH and np.mean(np.isnan(yi)) < NANTHRESH:
            if process:
                c = corr_coeff_nans(xi, yi)[0, 1]
            else:
                c = cov_nans(xi, yi)[0, 1]
            CC.append(c)
            Idxs.append(i * step + window //2)
    return np.array(CC), np.array(Idxs)

def correlate_signals_nans(s1, s2, normalised=1, mode='valid'):
    validIdxsX = ~np.isnan(s1)
    s1 = fill_in_nans(s1)
    s2 = fill_in_nans(s2)
    N = np.sum(validIdxsX)
    if normalised == 1:
        s1 = (s1 - np.mean(s1))
        s2 = (s2 - np.mean(s2))
    elif normalised == 2:
        s1 = signal.detrend(s1)
        s1 = (s1 - np.mean(s1)) / np.std(s1)
        s2 = signal.detrend(s2)
        s2 = (s2 - np.mean(s2)) / np.std(s2)
        
    corr = signal.correlate(s1, s2, mode)
    lags = signal.correlation_lags(len(s1), len(s2), mode)
    return lags, corr/N
    
def windowed_cross_corr(x, y, window, step):
    n = len(x)
    CC = []
    lags = []
    Idxs = []
    nwins = (n - 2 * window) // step
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    for i in range(nwins):
        xi = x[i * step + window //2 : i * step + (3 * window)//2]
        yi = y[i * step  : i * step + 2 * window]
        if np.mean(np.isnan(x[i * step  : i * step + 2 * window])) < NANTHRESH and np.mean(np.isnan(yi)) < NANTHRESH:
            lags, corr = correlate_signals_nans(xi, yi, mode='valid', normalised=1)
            CC.append(corr)
            Idxs.append(i * step + window)
    return np.array(lags) + window/2, np.array(CC), np.array(Idxs)


def windowed_cross_corr_corr_coeff(x, y, window, step, normalised=False):
    n = len(x)
    CC = []
    Coeffs = []
    lags = []
    Idxs = []
    nwins = (n - 2 * window) // step
    for i in range(nwins):
        xi = x[i * step + window //2 : i * step + (3 * window)//2]
        yi = y[i * step  : i * step + 2 * window]
        yii = y[i * step + window //2 : i * step + (3 * window)//2]
        if np.mean(np.isnan(xi)) < NANTHRESH and np.mean(np.isnan(yii)) < NANTHRESH:
            lags, corr = correlate_signals_nans(xi, yi, mode='valid')
            CC.append(corr)
            xii = x[i * step  : i * step + 2 * window]
            if normalised:
                c = corr_coeff_nans(xii, yi)[0, 1]
            else:
                c = cov_nans(xii, yi)[0, 1]
            Coeffs.append(c)
            Idxs.append(i * step + window)
    return np.array(lags) + window/2, np.array(CC), np.array(Coeffs), np.array(Idxs)

def windowed_cross_corr_max_pos(x, y, window, step):
    lags, corr0, Idxs = windowed_cross_corr(x, y, window, step)
    n = len(corr0)
    maxVal = np.zeros(n)
    maxPos = np.zeros(n)
    for i in range(n):
        a = np.abs(corr0[i])
        #idxs = np.argwhere(a >= st.scoreatpercentile(a, 95)).flatten()
        #idx = idxs[np.argmin(np.abs(idxs - window//2))]
        
        idx = np.argmax(a)
        maxPos[i] = lags[idx]
        maxVal[i] = corr0[i, idx]
    return maxPos, maxVal, Idxs

def windowed_cross_corr_max_pos_corr_coeff(x, y, window, step):
    lags, corr0, Coeffs, Idxs = windowed_cross_corr_corr_coeff(x, y, window, step)
    n = len(corr0)
    maxVal = np.zeros(n)
    maxPos = np.zeros(n)
    for i in range(n):
        idx = np.argmax(np.abs(corr0[i]))
        maxPos[i] = lags[idx]
        maxVal[i] = corr0[i, idx]
    return maxPos, maxVal, Coeffs, Idxs

def mark_reliable_frames(skeleton_length, ratio, val_min=-1, val_max=-1, blank=15, fps=8):
    nan_idxs = np.isnan(skeleton_length)
    skeleton_length = moving_average_nans(skeleton_length, 5)
    if val_min == -1:
        val_min = st.scoreatpercentile(skeleton_length[~nan_idxs], 50)
    if val_max == -1:
        val_max = st.scoreatpercentile(skeleton_length[~nan_idxs], 90)
    ratio_smooth = moving_average_nans(ratio, n=20)
    ratio80 = st.scoreatpercentile(ratio_smooth, 80)
    idxs = (skeleton_length > val_min) & (skeleton_length < val_max)
    print(val_min, val_max)
    lastBad = -2
    for i in range(len(idxs)):
        if not idxs[i]:
            lastBad = i
        elif lastBad + 1 == i:
            if (ratio_smooth[i] > ratio80) & (ratio_smooth[i] < ratio_smooth[i - 1]):
                lastBad = min([len(idxs), i + blank * fps])
                idxs[i : lastBad] = False
                print('erasing between ', i, 'and', lastBad)
    return idxs

def find_file(fname, dirList):
    for dir in dirList:
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name == fname:
                    fullfile = os.path.abspath(os.path.join(root, name))
                    return fullfile
    return None