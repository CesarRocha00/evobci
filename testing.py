import numpy as np
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

# Apply filters
notch(D, fs, w0, n_channels)
bandpass(D, fs, low, high, order, n_channels)

# Normalization
D[channel_names] = MinMaxScaler().fit_transform(D[channel_names])

# Total of events
event_index = D[D[labelID] == 1].index

# Total of samples
signal_size = D.index.size

# Custom fitness function
def my_custom_fitness(phenotype):
	# Copy original data to modify its labels
	X = D.copy()
	# Window segmentation vars
	wsize = int(round(phenotype[0]))
	wperc = round(phenotype[1], 2)
	wover = int(round(wsize * wperc))
	wstep = int(round(phenotype[2], 2) * (wsize - 20))
	nwind = wsize // wstep
	# print(wsize, wperc, wover, wstep, nwind)
	# return np.random.rand()
	param = {'wsize': wsize, 'wover': wover, 'label': labelID, 'channel': '1'}
	# Window segmentation (Training)
	X_train, y_train = make_fixed_windows(X, wsize, labelID, wover, nwind, wstep)
	# Map list of DataFrame to numpy arrays
	X_train = np.array([win[param['channel']].values for win in X_train])
	y_train = np.array(y_train)
	# Neural network vars
	N = wsize
	M = 32
	input_dim = N
	layer = [M, 1]
	activation = ['relu', 'sigmoid']
	optimizer = 'adam'
	loss = 'mean_squared_error'
	metrics = ['accuracy']
	# Neural network setup
	net = MLP_NN()
	net.build(input_dim, layer, activation, optimizer, loss, metrics)
	net.training(X_train, y_train, val_size=0.3, epochs=200)
	score = net.custom_validation(D, param)
	return score['acc']

# Parameters to optimize
# - window size
# - window overlap
# - number if windows per event
# - overlap of windows per event
# - low cut Hz
# - high cut Hz

sphere = lambda phenotype: sum(x**2 for x in phenotype)

# Genetic algorithm setup
alg = GeneticAlgorithm(100, 100, mutpb=-1, minmax='min')
alg.add_variable('x1', bounds=(-5.12, 5.12), precision=2)
alg.add_variable('x2', bounds=(-5.12, 5.12), precision=2)
# alg.add_variable('wsize', bounds=(50, 250), precision=0)
# alg.add_variable('wperc', bounds=(0.1, 0.9), precision=2)
# alg.add_variable('wstep', bounds=(0.01, 0.9), precision=2)
alg.set_fitness_func(sphere)
gbest, seed = alg.execute()
print('Seed:', seed)