import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('bright')
sns.set_style("whitegrid")

def boxplot(D, title, xlabel, ylabel):
	ax = sns.boxplot(data=D, showmeans=True, meanline=True)
	ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
	plt.show()

def lineplot(D, title, xlabel, ylabel, markers=False):
	ax = sns.lineplot(data=D['Accuracy'], markers=markers)
	ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
	# ax.set(title=title, xlabel=xlabel, ylabel=ylabel, yscale='log')
	plt.show()

D = pd.read_csv(sys.argv[1])

title = sys.argv[2]
xlabel = sys.argv[3]
ylabel = sys.argv[4]

# boxplot(D, title, xlabel, ylabel)

# xlabel = 'Execution'

lineplot(D, title, xlabel, ylabel, True)
