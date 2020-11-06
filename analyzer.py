import pandas as pd
from sys import argv
from pathlib import Path

D1 = pd.read_csv(argv[1])
channel_points = D1.columns[1:-2]
data = {'overlap': list(), 'points': list(), 'fitness': list()}
for i in D1.index:
	data['overlap'].append(D1.iloc[i]['overlap'])
	data['fitness'].append(D1.iloc[i]['fitness'])
	data['points'].append(D1.iloc[i][channel_points].sum())
D2 = pd.DataFrame(data)
print(D2)