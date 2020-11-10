import click
import numpy as np
import progressbar
import pandas as pd
from pathlib import Path
from neuralnet import MLP_NN
from evolalgo import GeneticAlgorithm
from sklearn.preprocessing import MinMaxScaler
from eeg_utils import notch, bandpass, split_train_test, extract_windows


class Experiment_1(object):
	"""
	Optimization of 
	- window size
	- window overlap
	- channel selection.

	* Training with consecutive overlaped windows
	* Validation with consecutive overlaped windows
	* Custom accuracy/fitness function
	"""

	# Global constants
	fs = 250
	w0 = 60
	low = 1
	high = 12
	order = 5
	metric = 'ACC'

	def __init__(self, kwargs):
		super(Experiment_1, self).__init__()
		self.kwargs = kwargs
		# Global variables
		self.D_train = None
		self.D_valid = None
		self.label_id = None
		self.channel_names = None
		self.channel_total = None
		self.best_val = None
		self.best_mod = None
		self.best_ind = None
		self.alg = None

	def data_preprocessing(self):
		# Data loading
		self.D_train = pd.read_csv(self.kwargs['inputfile'])
		# Get some info
		self.label_id = self.D_train.columns[-1]
		self.channel_names = self.D_train.columns[:-1]
		self.channel_total = self.channel_names.size
		# Apply filters
		notch(self.D_train, self.fs, self.w0, self.channel_total)
		bandpass(self.D_train, self.fs, self.low, self.high, self.order, self.channel_total)
		# Normalization
		self.D_train[self.channel_names] = MinMaxScaler().fit_transform(self.D_train[self.channel_names])
		# Split data into training and validation
		self.D_train, self.D_valid = split_train_test(self.D_train, self.label_id, self.kwargs['train_size'])

	def custom_metric(self, y_real, y_pred):
		score = {
			'TP': 0, 'FP': 0,
			'TN': 0, 'FN': 0,
			'LOST': 0,
			self.metric: 0.0
		}
		possibleFN, missingTP = False, True
		for real, pred in zip(y_real, y_pred):
			# Confusion matrix
			if real == 0 and pred == 0:
				score['TN'] += 1
			elif real == 0 and pred == 1:
				score['FP'] += 1
			elif real == 1 and pred == 1 and missingTP:
				score['TP'] += 1
				missingTP = False
			elif real == 1 and pred == 0 and missingTP:
				possibleFN = True
			else:
				score['LOST'] += 1
			# Check for FN and flag reset
			if real == 0:
				score['FN'] += 1 if possibleFN and missingTP else 0
				possibleFN, missingTP = False, True
		# Stats
		positives = score['TP'] + score['FN']
		negatives = score['TN'] + score['FP']
		score[self.metric] = 0.0
		if positives != 0 and negatives != 0:
			score[self.metric] = 0.5 * (score['TP'] / positives) + 0.5 * (score['TN'] / negatives)
		return score

	def custom_fitness(self, phenotype):
		# Make a copy of the continuous EEG
		X_train = self.D_train.copy()
		X_valid = self.D_valid.copy()
		# Extract the window segmentation vars
		wsize = phenotype['size']
		wover = phenotype['overlap']
		# Extract the list of selected channels
		channels = [name for name in self.channel_names if phenotype[name] == 1]
		channel_count = len(channels)
		# If channel list is empty, accuracy is zero
		if channel_count == 0:
			return 0.0
		# Window segmentation
		X_train, y_train = extract_windows(X_train, wsize, wover, self.label_id)
		X_valid, y_valid = extract_windows(X_valid, wsize, wover, self.label_id)
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
		history = model.train(X_train, y_train, val_size=0.3, epochs=self.kwargs['epochs'], verbose=0)
		y_pred = model.predict(X_valid)
		score = self.custom_metric(y_valid, y_pred)
		# Keep the best model
		if (self.best_val == None) or (score[self.metric] > self.best_val):
			self.best_val = score[self.metric]
			self.best_mod = model
		return score

	def run_algorithm(self):
		self.best_val = None
		self.best_mod = None
		self.best_ind = None
		# Initialize algorithm
		self.alg = GeneticAlgorithm(self.kwargs['pop_size'], self.kwargs['num_gen'], self.kwargs['cxpb'],
							   self.kwargs['cxtype'], mutpb=self.kwargs['mutpb'], minmax='max')
		# Add variables
		self.alg.add_variable('size', bounds=(50, 250), precision=0)
		self.alg.add_variable('overlap', bounds=(0.1, 0.9), precision=2)
		for name in self.channel_names:
			self.alg.add_variable(name, bounds=(0, 1), precision=0)
		# Set the fitness function
		self.alg.set_fitness_func(self.custom_fitness, self.metric)
		# Execute and get the best individual
		self.best_ind = self.alg.execute(verbose=0)
		
	def execute(self):
		self.data_preprocessing()
		widgets = ['Execution: ', progressbar.SimpleProgress(), ' [', progressbar.Percentage(), ']', 
		            progressbar.Bar(), progressbar.AbsoluteETA()]
		bar = progressbar.ProgressBar(widgets=widgets, max_value=self.kwargs['num_exe']).start()
		for i in range(self.kwargs['num_exe']):
			self.run_algorithm()
			self.save_data(i + 1)
			bar.update(i + 1)
		bar.finish()

	def save_data(self, id):
		# Prepare the directories
		path = self.prepare_directory()
		# Save the history of the current execution
		filename = path / f'execution_{id}.csv'
		self.alg.save(str(filename))
		# Save the best model of the current execution
		filename = filename.with_suffix('.h5')
		self.best_mod.save(str(filename))

	def prepare_directory(self):
		dir_parts = [self.kwargs['outputdir']]
		# Add the experiment name
		dir_parts.append(self.__class__.__name__)
		# Add the problem name (inputfile) keeping only the filename without extension
		dir_parts.append(Path(self.kwargs['inputfile']).stem)
		# Add the GA parameters (population and generations)
		dir_parts.append(f"P{self.kwargs['pop_size']}_G{self.kwargs['num_gen']}_E{self.kwargs['epochs']}")
		# Create directory of the experiment
		path = Path(*dir_parts)
		if not path.exists():
			path.mkdir(parents=True, exist_ok=True)
		return path


@click.command()
@click.argument('INPUTFILE', required=True)
@click.argument('OUTPUTDIR', required=True)
@click.option('-n', 'num_exe', type=click.IntRange(1, None), required=True, help='Number of executions.')
@click.option('-p', 'pop_size', type=click.IntRange(2, None), required=True, help='Size of the entire population.')
@click.option('-g', 'num_gen', type=click.IntRange(0, None), required=True, help='Number of generations to evolve.')
@click.option('-cp', 'cxpb', type=click.FloatRange(0.0, 0.9), default=0.9, show_default=True, help='Crossover probability.')
@click.option('-ct', 'cxtype', type=click.Choice(['npoint', 'binary'], case_sensitive=True), default='binary', show_default=True, help='Type of crossover to be applied.')
@click.option('-np', 'num_pts', type=click.IntRange(1, None), default=1, show_default=True, help='Number of crossover points.')
@click.option('-mp', 'mutpb', type=click.FloatRange(-1.0, 0.9), default=-1.0, show_default=True, help='Mutation probability. Values less than 0 means uniform mutation.')
@click.option('-e', 'epochs', type=click.IntRange(1, None), default=10, show_default=True, help='Epochs for ANN training.')
@click.option('-ts', 'train_size', type=click.FloatRange(0.1, 0.9), default=0.7, show_default=True, help='Split ratio for training and validation.')
def main(**kwargs):
	path = Path(kwargs['outputdir'])
	if not path.exists():
		print('Output directory does not exists!')
		return None
	expt = Experiment_1(kwargs)
	expt.execute()
		

if __name__ == '__main__':
	main()