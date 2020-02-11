import pandas as pd
from scipy.signal import iirnotch, filtfilt, butter

def notch(D, fs, w0, numCols, Q=30.0):
	columns = D.columns[:numCols]
	X = D[columns].values.T
	b, a = iirnotch(w0, Q, fs)
	X = filtfilt(b, a, X)
	D[columns] = X.T

def bandpass(D, fs, low, high, order, numCols):
	columns = D.columns[:numCols]
	X = D[columns].values.T
	nyq = fs * 0.5
	lowCut = low / nyq
	highCut = high / nyq
	b, a = butter(order, [lowCut, highCut], 'bandpass', output='ba')
	X = filtfilt(b, a, X)
	D[columns] = X.T