import pandas as pd
from eegtools import *
from neuralnet import MLP_NN
from evolalgo import GeneticAlgorithm
from sklearn.preprocessing import MinMaxScaler
# Filtering vars
fs = 250
w0 = 60
low = 1
high = 12
order = 5
n_channels = 8
labelID = 'Label'
# Data loading
D = pd.read_csv('Julio-Flores-23-Male_2020-02-12_1_Label.csv')
channel_names = D.columns[:-1]
notch(D, fs, w0, n_channels)
bandpass(D, fs, low, high, order, n_channels)
D[channel_names] = MinMaxScaler().fit_transform(D[channel_names])
# Window segmentation vars
wsize = max_window_size(D, labelID)
wperc = 0.5
wover = int(round(wsize * wperc))
# Window segmentatio
X, y = make_fixed_windows(D, wsize, labelID, wover, samples=2)
mins, maxs = wsize * 2, 0
for w in X:
	s = w.index.size
	if s < mins: mins = s
	if s > maxs: maxs = s
print(mins, maxs)
# X, y = make_fixed_windows(D, wsize, labelID, wover)
# Neural network vars
input_dim = wsize
layer = [input_dim * 2, input_dim, 1]
activation = ['relu', 'relu', 'sigmoid']
optimizer = 'adam'
loss = 'mean_squared_error'
metrics = ['accuracy']
# Neural network setup
# nnet = MLP_NN()
# nnet.build(input_dim, layer, activation, optimizer, loss, metrics)
# nnet.training(X_train, y_train, val_size=0.3, epochs=200)
# score = neuralnet.custom_validation(X, param)