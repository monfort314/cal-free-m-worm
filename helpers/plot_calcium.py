import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.optimize import curve_fit
from scipy.interpolate import splev, splrep, splprep
import scipy.stats as st
import pandas as pd
from helpers.skeletonise import * 
import os
import glob
from matplotlib import cm

def myexp(x, A, B):
    y = A * np.exp(B * x)
    return y

def find_curvature(X, Y, isGood):
    N, npts = X.shape 
    curv = np.zeros(shape=(N, npts))
    thetaArray = np.nan * np.zeros(shape=(N, npts))
    for ii in range(N):
        if isGood[ii]:
            x = X[ii, :] 
            y = Y[ii, :] 
            theta = np.zeros(npts)
            for i in range(1, npts - 1):
                theta[i] = np.arctan2((y[i + 1] - y[i - 1]) / 2, (x[i + 1] - x[i - 1]) / 2)    
            theta = np.unwrap(theta)
            theta = theta - np.mean(theta)
            curv[ii] = np.sum(np.abs(np.diff(theta)));
            thetaArray[ii] = theta
    return thetaArray, curv

def get_search_patch(frame, point, margin):
    x = int(point[0])
    y = int(point[1])
    height, width = frame.shape
    x1 = max([0, x - margin])
    x2 = min([width, x + margin])
    y1 = max([0, y - margin])
    y2 = min([height, y + margin])
    newPatch = frame[y1 : y2, x1 : x2]
    return newPatch, x1, x2, y1, y2

def plot_cells(dataFrame, skeletonArray, isSkeletonGood, fps, save, fname):
    N = skeletonArray.shape[0]
    T = N / fps
    t = np.linspace(0, T, N)
    icells = (len(dataFrame.columns) - 1) // 2
    bodyPts = np.arange(skeletonArray.shape[1])
    thetaArray, curv = findCurvature(skeletonArray[:, :, 0], skeletonArray[:, :, 1], isSkeletonGood)
    ratio = np.zeros(shape=(N, icells))
    for j in range(icells):
        ds = dataFrame['cell' + str(j + 1) + '_green']
        #parameters, covariance = curve_fit(myexp, t, ds)
        #subtractedGreen = ds - myexp(t, parameters[0], parameters[1])
        subtractedGreen = ds

        ds = dataFrame['cell' + str(j + 1) + '_red']
        #parameters, covariance = curve_fit(myexp, t, ds)
        #subtractedRed = ds - myexp(t, parameters[0], parameters[1])
        subtractedRed = ds

        ratio[:, j] = subtractedGreen / (subtractedRed + 1e-6)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].pcolor(t, bodyPts, thetaArray.T)
    axs[1].plot(t, ratio)
    if save: 
        plt.savefig(fname)
    return thetaArray, ratio, t


def extract_signal(dataFrame, GCamp, tagRFP, margin=10):
    icells = (len(dataFrame.columns) - 1) // 2
    N = GCamp.shape[0]
    data = {'timepoints': []}
    for i in range(N):
        frameGreen = GCamp[i]
        frameRed = tagRFP[i]
        data['timepoints'].append(i)
        for j in range(icells):
            if i == 0:
               data['cell' + str(j + 1) + '_green'] = []
               data['cell' + str(j + 1) + '_red'] = []
            score = dataFrame['score' + str(j + 1)].iloc[i]
            if score >= 0.5:    
                coord = dataFrame['cell' + str(j + 1)]
                point = coord.iloc[i]
                patchGreen, x1, x2, y1, y2 = get_search_patch(frameGreen, point, margin)
                patchRed, x1, x2, y1, y2 = get_search_patch(frameRed, point, margin)
                valGreen = np.sum(patchGreen)
                valRed = np.sum(patchRed)
            else:
                valGreen = np.nan
                valRed = np.nan

            data['cell' + str(j + 1) + '_green'].append(valGreen)
            data['cell' + str(j + 1) + '_red'].append(valRed)
    dataFrameCa = pd.DataFrame(data=data)
    return dataFrameCa


def smooth_spline_nans(x, y):
    N = len(x)
    nans = np.isnan(x)
    t = np.linspace(0, 1, N)
    treal = t[~nans]
    xreal = x[~nans]
    yreal = y[~nans]
    
    tck, u = splprep([treal, xreal, yreal], s=0)
    out = splev(t, tck)
    xnew = out[1]
    ynew = out[2]
    xnew[nans] = np.nan
    ynew[nans] = np.nan
    # plt.figure()
    # plt.plot(x, y, '-o')
    # plt.plot(xnew, ynew, '-o')
    # plt.show()
    return xnew, ynew

def string2matrix(word):
    words = word[1:-2].split(' ')
    newList = []
    for word in words:
        try:
            newList.append(int(word))
        except:
            pass
    return np.array(newList)

def interpolateIdxs(x, idxs):
    
    nans = np.where(np.isnan(x))[0]
    N = len(x)

    idxsBool = np.ones(N, dtype=bool)
    idxsBool[idxs] = False
    idxsBool[nans] = False

    t = np.arange(len(x))
    treal = t[idxsBool]
    xreal = x[idxsBool]
    spl = splrep(treal, xreal)
    xinter = splev(t, spl)
    xinter[nans] = np.nan
    return xinter

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

def plot_ratio_velocity(t, Ratio, Vel, neurons, fname_save, normalise=False):
    fig, axs = plt.subplots(nrows=len(neurons), sharex=True)
    for i, neuron in enumerate(neurons):
        if len(neurons) > 1:
            ax = axs[i]
        else:
            ax = axs
        ratio = Ratio[i]
        if normalise:
            r = ratio[~np.isnan(ratio)]
            ratio20 = st.scoreatpercentile(r, 20)
            ratio = (ratio - ratio20) / ratio20
        plotr, = ax.plot(t, ratio, 'k')
        ax.set_title(neuron)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.set_ylim(-0.3, 4)
        ax2 = ax.twinx()
        plotTang, = ax2.plot(t, Vel[i, :, 0], c='g')
        plotNorm, = ax2.plot(t, Vel[i, :, 1], c='c') 
        ax2.axhline(y=0, linestyle='--', alpha=0.3)
        ax2.spines['top'].set_visible(False) 
        a = 350
        ax2.set_ylim(-a, a)
        if i == len(neurons) - 1:
            ax.legend([plotr, plotTang, plotNorm], ['$\\Delta F / F_{0}$', '$v_{t}$', '$v_{n}$'])
            ax.set_xlabel('Time (sec)')
    fig.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    fig.savefig(fname_save + '_ca_vel.png')   
    
def ca_activity_velocity(skeleton, velocity_units, path, fname, fname_save, section, height, fps, pixToUm):
    # plot the activity of the each TRN and the two components of the velocity
    X = skeleton[:, :, 0]
    Y = skeleton[:, :, 1]
    npts = skeleton.shape[1]
    xls = os.path.join(path, fname.replace('-', '_') + '.xlsx')
    masterPath = os.path.split(path)[0]
    
    df = pd.read_excel(xls, sheet_name=None)
    neurons = list(df.keys())
    eps = 2
    N = 960
    Ratio = np.nan * np.zeros(shape=(len(neurons), N))
    VelocityNrn = np.nan * np.zeros(shape=(len(neurons), N, 2))
    Vel = np.nan * np.zeros(shape=(len(neurons), N, 2))
    for i, neuron in enumerate(neurons):
        print(neuron)
        data = df[neuron]
        N = max([N, len(data['x'])])
        t = np.linspace(0, N / fps, N)
        x = data['x']
        y = data['y']
        dx = np.concatenate([np.diff(x), [0]])
        y = height - y
        dy = np.concatenate([np.diff(y), [0]])
        for j in range(N):
            if not np.isnan(X[j] + x[j]).all():
                d = np.sqrt((X[j] - x[j])**2 + (Y[j] - y[j])**2)
                l = j
                idx = np.argmin(d)
                #print(mini,maxi)
                (eT, eN) = velocity_units[j, idx]
                v = np.array([dx[j], dy[j]]) * pixToUm * fps
                (vel_t, vel_n) = vectorDecomposition2(v, eT, eN)
                VelocityNrn[i, j] = (vel_t, vel_n)
        signalRed = data['red']
        signalGreen = data['green']
        
        signalRed = moving_average_nans(signalRed, 3)
        signalGreen = moving_average_nans(signalGreen, 3)

        ratio = signalGreen / (signalRed + eps)
        Ratio[i] = ratio

        Vel[i, :, 0] = moving_average_nans(VelocityNrn[i, :, 0], 5)
        Vel[i, :, 1] = moving_average_nans(VelocityNrn[i, :, 1], 5)
    plot_ratio_velocity(t, Ratio, Vel, neurons, fname_save, False)
    plot_ratio_velocity(t, Ratio, Vel, neurons, fname_save + '_norm', True)
    
    return Ratio, VelocityNrn, neurons
