import sys
import click
import numpy as np
import pandas as pd
from datetime import datetime
from eegtools import *
from neuralnet import MLP_NN
from evolalgo import GeneticAlgorithm
from sklearn.preprocessing import MinMaxScaler

# Global constants
fs = 250
w0 = 60
low = 1
high = 12
order = 5
# Global variables
D_train = None
D_valid = None
label_id = None
channel_names = None
channel_total = None
epochs = 32

# Custom accuracy
def my_accuracy(y_real, y_pred):
	TP = FP = TN = FN = 0
	lost = 0
	possibleFN, missingTP = False, True
	for real, pred in zip(y_real, y_pred):
		# Confusion matrix
		if real == 0 and pred == 0:
			TN += 1
		elif real == 0 and pred == 1:
			FP += 1
		elif real == 1 and pred == 1 and missingTP:
			TP += 1
			missingTP = False
		elif real == 1 and pred == 0 and missingTP:
			possibleFN = True
		else:
			lost += 1
		# Check for FN and flag reset
		if real == 0:
			FN += 1 if possibleFN and missingTP else 0
			possibleFN, missingTP = False, True
	# Stats
	positives = TP + FN
	negatives = TN + FP
	accuracy = 0.0
	if positives != 0 and negatives != 0:
		accuracy = 0.5 * (TP / positives) + 0.5 * (TN / negatives)
	score = {
		'tp': TP, 'fp': FP,
		'tn': TN, 'fn': FN,
		'accuracy': accuracy
	}
	return score

def count_ones(y_list):
	counters = dict()
	count = 0
	mark = False
	for i in y_list:
		if i == 1:
			count += 1
		elif i == 0 and count > 0:
			if count in counters.keys():
				counters[count] += 1
			else:
				counters[count] = 1
			count = 0
	return counters

# Custom fitness function
def my_fitness(phenotype):
	# Make a copy of the continuous EEG
	X_train = D_train.copy()
	X_valid = D_valid.copy()
	# Extract the window segmentation vars
	wsize = phenotype[0]
	wover = phenotype[1]
	wpadd = phenotype[2]
	wstep = phenotype[3]
	# Extract the list of selected channels
	channels = [channel_names[i] for i in range(channel_total) if phenotype[i + 4] == 1]
	channel_count = len(channels)
	# If channel list is empty, accuracy is zero
	if channel_count == 0:
		return 0.0
	# Window segmentation
	X_train, y_train = extract_windows(X_train, wsize, wover, label_id, True, wpadd, wstep)
	X_valid, y_valid = extract_windows(X_valid, wsize, wover, label_id)
	# Map list of DataFrames to numpy arrays
	X_train = np.array([win[channels].values.T.ravel() for win in X_train])
	y_train = np.array(y_train)
	X_valid = np.array([win[channels].values.T.ravel() for win in X_valid])
	y_valid = np.array(y_valid)
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
	metrics = None
	# Neural network setup
	model = MLP_NN()
	model.build(input_dim, layer, activation, optimizer, loss, metrics)
	history = model.train(X_train, y_train, val_size=0.3, epochs=epochs, verbose=0)
	y_pred = model.predict(X_valid)
	score = my_accuracy(y_valid, y_pred)
	accuracy = score['accuracy']
	return accuracy
	# model.save('./last_model.h5')

def run_algorithm(kwargs):
	ga = GeneticAlgorithm(pop_size=kwargs['pop_size'], num_gen=kwargs['num_gen'], cxpb=kwargs['cxpb'], cxtype=kwargs['cxtype'], mutpb=kwargs['mutpb'], minmax='max')
	ga.add_variable('wsize', bounds=(50, 250), precision=0)
	ga.add_variable('wover', bounds=(0.1, 0.9), precision=2)
	ga.add_variable('wpadd', bounds=(0.1, 0.9), precision=1)
	ga.add_variable('wstep', bounds=(0.1, 0.9), precision=1)
	for name in channel_names:
		ga.add_variable(name, bounds=(0, 1), precision=0)
	ga.set_fitness_func(my_fitness)
	gbest, seed = ga.execute()

def execute(kwargs):
	for i in range(kwargs['num_exe']):
		run_algorithm(kwargs)

def data_preprocessing(kwargs):
	global D_train, D_valid
	global label_id, channel_names, channel_total
	# Data loading
	D_train = pd.read_csv(kwargs['inputfile'])
	# Get some info
	label_id = D_train.columns[-1]
	channel_names = D_train.columns[:-1]
	channel_total = channel_names.size
	# Apply filters
	notch(D_train, fs, w0, channel_total)
	bandpass(D_train, fs, low, high, order, channel_total)
	# Normalization
	D_train[channel_names] = MinMaxScaler().fit_transform(D_train[channel_names])
	# Split data into training and validation
	D_train, D_valid = split_training_validation(D_train, label_id, kwargs['split_ratio'])

@click.command()
@click.argument('INPUTFILE', required=True)
@click.option('-n', 'num_exe', type=click.IntRange(1, None), required=True, help='Number of executions.')
@click.option('-p', 'pop_size', type=click.IntRange(2, None), required=True, help='Size of the entire population.')
@click.option('-g', 'num_gen', type=click.IntRange(0, None), required=True, help='Number of generations to evolve.')
@click.option('-cp', 'cxpb', type=click.FloatRange(0.0, 0.9), default=0.9, show_default=True, help='Crossover probability.')
@click.option('-ct', 'cxtype', type=click.Choice(['npoint', 'binary'], case_sensitive=True), default='binary', show_default=True, help='Type of crossover to be applied.')
@click.option('-np', 'num_pts', type=click.IntRange(1, None), default=1, show_default=True, help='Number of crossover points.')
@click.option('-mp', 'mutpb', type=click.FloatRange(-1.0, 0.9), default=-1.0, show_default=True, help='Mutation probability. Values less than 0 mean uniform mutation.')
@click.option('-e', 'epochs', type=click.IntRange(1, None), default=10, show_default=True, help='Epochs for ANN training.')
@click.option('-sr', 'split_ratio', type=click.FloatRange(0.1, 0.9), default=0.7, show_default=True, help='Split ratio for training and validation.')
def main(**kwargs):
	# click.echo(kwargs)
	data_preprocessing(kwargs)
	run_algorithm(kwargs)
	# my_fitness([71, 0.29, 0.1, 0.1, 1, 1, 0, 1, 1, 1, 0, 0])

if __name__ == '__main__':
	main()