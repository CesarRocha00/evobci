import click
import numpy as np
import pandas as pd
from pathlib import Path
from neuralnet import MLP_NN
from sklearn.preprocessing import MinMaxScaler
from eeg_utils import notch, bandpass, extract_windows

def compute_matrix(y_real, y_pred):
    matrix = {
        'TP': 0, 'FP': 0,
        'TN': 0, 'FN': 0,
        'LOST': 0
    }
    possibleFN, missingTP = False, True
    for real, pred in zip(y_real, y_pred):
        # Confusion matrix
        if real == 0 and pred == 0:
            matrix['TN'] += 1
        elif real == 0 and pred == 1:
            matrix['FP'] += 1
        elif real == 1 and pred == 1 and missingTP:
            matrix['TP'] += 1
            missingTP = False
        elif real == 1 and pred == 0 and missingTP:
            possibleFN = True
        else:
            matrix['LOST'] += 1
        # Check for FN and flag reset
        if real == 0:
            matrix['FN'] += 1 if possibleFN and missingTP else 0
            possibleFN, missingTP = False, True
    # Stats
    try:
        matrix['PPV'] = matrix['TP'] / (matrix['TP'] + matrix['FP'])
        matrix['TPR'] = matrix['TP'] / (matrix['TP'] + matrix['FN'])
        matrix['F1'] = 2 * (matrix['PPV'] * matrix['TPR']) / (matrix['PPV'] + matrix['TPR'])
    except ZeroDivisionError:
        pass
    return matrix

class Experiment_2(object):
    """
    Testing of trained models
    """

    # Global constants
    fs = 250
    w0 = 60
    low = 1
    high = 12
    order = 5
    wsize = 250

    def __init__(self, kwargs):
        super(Experiment_2, self).__init__()
        self.kwargs = kwargs
        self.model = None
        self.wover = None
        self.D_test = None
        self.label_id = None
        self.channel_total = None
        self.channel_names = None
        self.variable_names = None

    def data_preprocessing(self):
        self.D_test = pd.read_csv(self.kwargs['eegfile'])
        # Get some info
        self.label_id = self.D_test.columns[-1]
        self.channel_names = self.D_test.columns[:-1]
        self.variable_names = [f'{name}_{i + 1}' for name in self.channel_names for i in range(self.wsize)]
        self.channel_total = self.channel_names.size
        # Apply filters
        notch(self.D_test, self.fs, self.w0, self.channel_total)
        bandpass(self.D_test, self.fs, self.low, self.high, self.order, self.channel_total)
        # Normalization
        self.D_test[self.channel_names] = MinMaxScaler().fit_transform(self.D_test[self.channel_names])
        
    def execute(self):
        self.data_preprocessing()
        # Model setup
        self.model = MLP_NN()
        self.model.load(self.kwargs['model'])
        # Load
        path = Path(self.kwargs['model'])
        path = path.with_suffix('.csv')
        S = pd.read_csv(path)
        best = S.tail(1)
		# List of active channels
        channels = [name for name in self.channel_names if best[name][199] == 1]
        # List if variable names
        variable_names = [f'{name}_{i + 1}' for name in channels for i in range(self.wsize)]
        # List of active/inactive positios
        positions = np.array([best[name][199] for name in variable_names])
        # Check for inactive positions
        inactives = np.where(positions == 0)[0]
        # Window extraction as pandas DataFrame
        X_test, y_test = extract_windows(self.D_test, self.wsize, best['overlap'][199], self.label_id)
        # Map list of DataFrames to numpy arrays
        X_test = np.array([win[channels].T.values.ravel() for win in X_test])
        y_test = np.array(y_test)
        # Delete inactive positios
        X_test = np.delete(X_test, inactives, axis=1)
        # Make predictions
        y_pred = self.model.predict(X_test)
        # Compute metrics
        matrix = compute_matrix(y_test, y_pred)
        M = pd.Series(matrix)
        print('------------------------------')
        print(f'Training: {path.parts[1]}')
        print(f'Best model: {path.stem}')
        print(f'Channels: {channels}')
        print(f'Total features: {positions.sum()}')
        path = Path(self.kwargs['eegfile'])
        print(f'Testing: {path.stem}')
        print('------------------------------')
        print(M)
        print('------------------------------')

@click.command()
@click.argument('MODEL', type=click.Path(exists=True, dir_okay=False), required=True)
@click.argument('EEGFILE', type=click.Path(exists=True, dir_okay=False), required=True)
def main(**kwargs):
    expt = Experiment_2(kwargs)
    expt.execute()

main()
