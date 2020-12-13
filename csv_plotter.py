import click
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

sns.set_palette('bright')
sns.set_style("whitegrid")

def load_data(inputfile, xcolumn, ycolumn):
	# Generate a list of Path objects by searching CSV files
	filepath = [Path(file) for file in inputfile]
	# Iterate through each path in list and open it as a DataFrame
	data_frames = dict()
	for path in filepath:
		df = pd.read_csv(path)
		if (xcolumn == 'index' or ycolumn == 'index') and 'index' not in df.columns:
			df['index'] = df.index + 1
		try:
			data_frames[path.stem] = df[[xcolumn, ycolumn]]
		except KeyError:
			print(f'{path} does not have {[ col for col in [xcolumn, ycolumn] if col not in df.columns ]} column names!')
	return data_frames

@click.group()
def cli():
	pass

@cli.command()
@click.argument('INPUTFILE', type=click.Path(exists=True, dir_okay=False), nargs=-1, required=True)
@click.option('-x', type=click.Tuple([str, str]), required=True, help='X axis: CSV column name and label.')
@click.option('-y', type=click.Tuple([str, str]), required=True, help='Y axis: CSV column name and label.')
@click.option('-t', type=click.STRING, help='Plot title.')
def lineplot(**kwargs):
	data_frames = load_data(kwargs['inputfile'], kwargs['x'][0], kwargs['y'][0])
	ax = None
	for filename, df in data_frames.items():
		ax = sns.lineplot(x=kwargs['x'][0], y=kwargs['y'][0], data=df, label=filename)
	ax.set(title=kwargs['t'], xlabel=kwargs['x'][1], ylabel=kwargs['y'][1])
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	cli()