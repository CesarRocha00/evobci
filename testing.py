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

# Data loading
D_train = pd.read_csv(sys.argv[1])
D_valid = pd.read_csv(sys.argv[2])

# Get some info
channel_names = D_train.columns[:-1]
total_channels = channel_names.size
label_id = D_train.columns[-1]

# Apply filters
notch(D_train, fs, w0, total_channels)
bandpass(D_train, fs, low, high, order, total_channels)
notch(D_valid, fs, w0, total_channels)
bandpass(D_valid, fs, low, high, order, total_channels)

# Normalization
D_train[channel_names] = MinMaxScaler().fit_transform(D_train[channel_names])
D_valid[channel_names] = MinMaxScaler().fit_transform(D_valid[channel_names])

# Split data into training and validation
D_train, D_optim = split_training_validation(D_train, label_id, 0.7)

# Variable to check the best accuracy and save the model
gbest_acc = 0.0

# Custom fitness function
def my_fitness(phenotype, epochs=10):
	global gbest_acc
	# Copy original data to modify its labels
	X_train = D_train.copy()
	# Window segmentation vars
	wsize = phenotype[0]
	wover = phenotype[1]
	wpadd = phenotype[2]
	wstep = phenotype[3]
	# Channel list
	channels = [channel_names[i] for i in range(total_channels) if phenotype[i + 4] == 1]
	selected_channels = len(channels)
	# If channel list is empty, acurracy is zero
	if total_channels == 0:
		return 0.0
	# Variables to perform a custom validation
	wover_val = int(round(wover * wsize))
	# Window segmentation (Training)
	X_train, y_train = extract_windows(X_train, wsize, wover, label_id, True, wpadd, wstep)
	# Map list of DataFrame to numpy arrays
	X_train = np.array([win[channels].values.T.ravel() for win in X_train])
	y_train = np.array(y_train)
	# Neural network structure
	input_dim = X_train[0].size
	output_dim = 1
	# Two hidden layers
	ratio_io = int((input_dim / output_dim) ** (1 / 3))
	hidden_1 = output_dim * (ratio_io ** 2)
	hidden_2 = output_dim * ratio_io
	layer = [hidden_1, hidden_2, output_dim]
	activation = ['relu', 'relu', 'sigmoid']
	# Optimizer and metrics
	optimizer = 'adam'
	loss = 'mse'
	metrics = ['accuracy']
	# Neural network setup
	model = MLP_NN()
	model.build(input_dim, layer, activation, optimizer, loss, metrics)
	model.train(X_train, y_train, val_size=0.3, epochs=epochs, verbose=0)
	score = model.validation(D_optim, wsize, wover, channels, label_id)
	# Accuracy type
	accuracy  = score['accuracy']
	# accuracy = 0.8 * score['accuracy'] + 0.2 * (1 - selected_channels / total_channels)
	# Model backup
	if accuracy > gbest_acc:
		model.save('./models/best')
		gbest_acc = accuracy
	# Return the accuracy as fitness
	return accuracy

# Sphere function for testing
sphere = lambda phenotype: sum(x ** 2 for x in phenotype)

# Parameters to optimize
# - window size
# - window overlap
# - window step for resampling
# - window padding for resampling
# - list of channels

# Genetic algorithm setup
def run_ga():
	alg = GeneticAlgorithm(20, 30, mutpb=-1, minmax='max', seed=None)
	alg.add_variable('wsize', bounds=(50, 250), precision=0)
	alg.add_variable('wover', bounds=(0.1, 0.9), precision=2)
	alg.add_variable('wpadd', bounds=(0.1, 0.9), precision=1)
	alg.add_variable('wstep', bounds=(0.1, 0.9), precision=1)
	for name in channel_names:
		alg.add_variable(name, bounds=(0, 1), precision=0)
	alg.set_fitness_func(my_fitness)
	gbest, seed = alg.execute()
	print('Seed: {}'.format(seed))
	return gbest

# Epoch sintonization
def run_epoch():
	for i in range(31):
		for e in [8, 16, 32, 64, 128]:
			print(my_fitness([51, 0.33, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1], e), end=',')
		print()

# Test specific cases
def run_case():
	for i in range(2):
		print(my_fitness([51, 0.33, 0.1, 0.1, 1, 1, 0, 0, 0, 0, 0, 0], 10))

# Run GA
gbest = run_ga()
# Load the best model
new_model = MLP_NN()
new_model.load('./models/best')
# Get parameters from gbest
wsize = gbest.phenotype[0]
wover = gbest.phenotype[1]
channels = [channel_names[i] for i in range(total_channels) if gbest.phenotype[i + 4] == 1]
# Perform the validation
score = new_model.validation(D_valid, wsize, wover, channels, label_id)
print('GA accuracy:', gbest.fitness)
print('Verification:', score['accuracy'])
