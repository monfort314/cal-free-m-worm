from scipy import signal
import numpy as np


def fillNans(x):
    xi = np.arange(len(x))
    mask = np.isfinite(x)
    xfiltered = np.interp(xi, xi[mask], x[mask])
    return xfiltered

def corrCoeffNans(s1, s2, process=True):
    validIdxsX = ~np.isnan(s1)
    s1 = fillNans(s1)
    s2 = fillNans(s2)
    return np.corrcoef(s1, s2)
    
def SignalStatistics(ratio, velocity, neurons, maxshift):
	coherence = np.zeros(shape=(len(neurons), 3))
	shift = np.zeros(shape=(len(neurons), 3))
	correlation = np.zeros(shape=(len(neurons), 3))
	if len(ratio.shape) == 1:
		ratio = np.reshape(ratio, newshape=(1, -1))
		velocity = np.reshape(velocity, newshape=(1, -1, 2))
	Vel3 = np.zeros(shape=(velocity.shape[0], velocity.shape[1], 3))
	Vel3[:, :, :2] = velocity
	Vel3[:, :, -1] = np.sqrt(velocity[:, :, 0]**2 + velocity[:, :, 1]**2)
	N = ratio.shape[1]
	for i, nrn in enumerate(neurons):
		x = ratio[i]
		validIdxsX = ~np.isnan(x)
		x = fillNans(x)
		x = signal.detrend(x)
		x = (x - np.mean(x)) / np.std(x)
		htx = signal.hilbert(x)
		insfreqx = np.angle(htx)

		for j in range(3):
			y = Vel3[i, :, j]
			validIdxsY = ~np.isnan(y)
			y = fillNans(y)
			y = signal.detrend(y)
			y = (y - np.mean(y)) / np.std(y)
			hty = signal.hilbert(y)
			insfreqy = np.angle(hty)
			validIdxs = validIdxsX & validIdxsY
			coherency = np.exp(-1j * (insfreqx[validIdxs] - insfreqy[validIdxs]))
			coherence[i, j] = abs(np.mean(coherency))
			corr = signal.correlate(x, y)[N - maxshift : N + maxshift]
			lags = signal.correlation_lags(len(x), len(y))[N - maxshift : N + maxshift]
			imax = np.argmax(corr)
			shift[i, j] = lags[imax]
			correlation[i, j] = corr[imax] / N
	return coherence, correlation, shift