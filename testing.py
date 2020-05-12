import numpy as np
import pandas as pd
from eegtools import *
from neuralnet import MLP_NN
from datetime import datetime
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
D = pd.read_csv('../../../../Desktop/Sessions/Julio-Flores-23-Male_2020-02-12_1_Label.csv')
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
def my_fitness(phenotype):
	# Copy original data to modify its labels
	X = D.copy()
	# Window segmentation vars
	wsize = phenotype[0]
	wover = phenotype[1]
	wstep = phenotype[2]
	wpadding = phenotype[3]
	# Variables to perform a custom validation
	wover_val = int(round(wover * wsize))
	param = {'wsize': wsize, 'wover': wover_val, 'label': labelID, 'channel': '1'}
	# Window segmentation (Training)
	X_train, y_train = extract_windows(X, wsize, labelID, wover, True, wstep, wpadding)
	# Map list of DataFrame to numpy arrays
	X_train = np.array([win[param['channel']].values for win in X_train])
	y_train = np.array(y_train)
	# Neural network structure
	input_dim = wsize
	output_dim = 1
	# # One hidden layer
	# hidden_1 = int((input_dim * output_dim) ** (1 / 2))
	# layer = [hidden_1, output_dim]
	# activation = ['relu', 'sigmoid']
	# Two hidden layers
	ratio_io = int((input_dim / output_dim) ** (1 / 3))
	hidden_1 = output_dim * (ratio_io ** 2)
	hidden_2 = output_dim * ratio_io
	layer = [hidden_1, hidden_2, output_dim]
	activation = ['relu', 'relu', 'sigmoid']
	# Optimizer and metrics
	optimizer = 'adam'
	loss = 'mean_squared_error'
	metrics = ['accuracy']
	# Neural network setup
	net = MLP_NN()
	net.build(input_dim, layer, activation, optimizer, loss, metrics)
	net.training(X_train, y_train, val_size=0.3, epochs=100)
	score = net.my_validation(D, param)
	return score['acc']

# Sphere function for testing
sphere = lambda phenotype: sum(x ** 2 for x in phenotype)

# Parameters to optimize
# - window size
# - window overlap
# - window step for resampling
# - window padding for resampling
# - low cut Hz
# - high cut Hz

# Genetic algorithm setup
# alg = GeneticAlgorithm(2, 0, mutpb=-1, minmax='max', seed=0)
# alg.add_variable('wsize', bounds=(50, 250), precision=0)
# alg.add_variable('wover', bounds=(0.1, 0.9), precision=2)
# alg.add_variable('wstep', bounds=(0.1, 0.9), precision=1)
# alg.add_variable('wpadding', bounds=(0.1, 0.9), precision=1)
# alg.set_fitness_func(my_fitness)
# gbest, seed = alg.execute()
# print('Seed:', seed)

print(my_fitness([50, 0.33, 0.1, 0.1]))