import click
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

wsize = 250
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'P1', 'P2', 'O1', 'O2']

def load_data(inputpath):
	# Generate a list of Path objects by searching CSV files
	path = Path(inputpath)
	filepath = list(path.glob('*.csv'))
	# Iterate through each path in list and open it as a DataFrame
	file_id = list()
	elapsed_time = list()
	experiment = pd.DataFrame()
	for path in filepath:
		df = pd.read_csv(path)
		gbest = df.tail(1)
		file_number = path.stem.split('_')[1]
		file_id.append(int(file_number))
		seconds = df['elapsed'].sum()
		elapsed_time.append(str(timedelta(seconds=seconds)))
		experiment = experiment.append(gbest, ignore_index=True)
	experiment.insert(0, 'file_id', file_id)
	experiment['elapsed'] = elapsed_time
	return experiment

def compute_stats(kwargs):
	experiment = load_data(kwargs['inputpath'])
	# Summary of channel selection
	for name in channels:
		positions = [f'{name}_{i + 1}' for i in range(wsize)]
		experiment[name] = experiment[positions].sum(axis=1) * experiment[name]
		experiment.drop(columns=positions, inplace=True)
	total_points = experiment[channels].sum(axis=1)
	pos = int(np.where(experiment.columns == 'TP')[0][0])
	experiment.insert(pos, 'points', total_points)
	cols_to_int = experiment.columns[2:-2]
	experiment[cols_to_int] = experiment[cols_to_int].applymap(int)
	# Sort by file id or fitness value
	col_key = ['fitness', 'points'] if kwargs['sort'] else 'file_id'
	order = [True, False] if kwargs['sort'] else True
	experiment.sort_values(by=col_key, ascending=order, inplace=True, ignore_index=True)
	# Display mode
	print('# Best individual per execution')
	if kwargs['display'] == 'csv':
		print(experiment.to_csv(index=False))
	elif kwargs['display'] == 'latex':
		print(experiment.to_latex(index=False))
	else:
		print(experiment)
	# Compute mean, std, median
	stats = pd.Series(dtype='float64')
	stats['std'] = experiment['fitness'].std()
	stats['mean'] = experiment['fitness'].mean()
	stats['median'] = experiment['fitness'].median()
	print('# Statistics')
	print(stats)

@click.command()
@click.argument('INPUTPATH', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('-d', 'display', type=click.Choice(['plain', 'csv', 'latex'], case_sensitive=True), default='plain', show_default=True, help='Display mode. Plain DataFrame or LaTeX table.')
@click.option('-s', 'sort', is_flag=True, default=False, show_default=True, help='Sorting mode. Sort the DataFrame by fitness value.')
def main(**kwargs):
	compute_stats(kwargs)

if __name__ == '__main__':
	main()