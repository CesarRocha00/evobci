import click
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

sns.set_palette('bright')
sns.set_style("whitegrid")

def load_data(filepaths, x_col, y_col):
	data_frames = dict()
	for path in filepaths:
		file = Path(path)
		df = pd.read_csv(file)
		if (x_col == 'index' or y_col == 'index') and 'index' not in df.columns:
			df['index'] = df.index + 1
		try:
			data_frames[file.stem] = df[[x_col, y_col]]
		except KeyError:
			print(f'{path} does not have {[ col for col in [x_col, y_col] if col not in df.columns ]} column names!')
	return data_frames

@click.group()
def cli():
	pass

@cli.command()
@click.argument('INPUTFILE', type=click.Path(exists=True, dir_okay=False), nargs=-1, required=True)
@click.option('-x', 'x_col', type=click.STRING, required=True, help='Column name of X axis.')
@click.option('-y', 'y_col', type=click.STRING, required=True, help='Column name of Y axis.')
@click.option('-xl', 'x_lab', type=click.STRING, help='Label of X axis.')
@click.option('-yl', 'y_lab', type=click.STRING, help='Label of Y axis.')
def lineplot(**kwargs):
	data_frames = load_data(kwargs['inputfile'], kwargs['x_col'], kwargs['y_col'])
	for file, df in data_frames.items():
		ax = sns.lineplot(x=kwargs['x_col'], y=kwargs['y_col'], data=df, label=file)
		ax.set(xlabel=kwargs['x_lab'], ylabel=kwargs['y_lab'])
	plt.tight_layout()
	plt.show()

@cli.command()
def boxplot(D, title, xlabel, ylabel):
	ax = sns.boxplot(data=D, showmeans=True, meanline=True)
	ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
	plt.show()

if __name__ == '__main__':
	cli()