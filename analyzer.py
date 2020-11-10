import pandas as pd
from sys import argv
from pathlib import Path

D = pd.read_csv(argv[1])
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'P1', 'P2', 'O1', 'O2']
wsize = 250
S = pd.DataFrame()
S['overlap'] = D['overlap']
for name in channels:
	positions = [f'{name}_{i + 1}' for i in range(wsize)]
	S[name] = D[positions].sum(axis=1)
S['points'] = S[channels].sum(axis=1)
S['fitness'] = D['fitness']
S['TP'] = D['TP']
S['FP'] = D['FP']
S['TN'] = D['TN']
S['FN'] = D['FN']
# S['LOST'] = D['lost']
S['TPR'] = S['TP'] / S[['TP','FN']].sum(axis=1)
S['TNR'] = S['TN'] / S[['TN','FP']].sum(axis=1)
S['PPV'] = S['TP'] / S[['TP','FP']].sum(axis=1)
S['NPV'] = S['TN'] / S[['TN','FN']].sum(axis=1)
S['FNR'] = 1 - S['TPR']
S['FPR'] = 1 - S['TNR']
S['FDR'] = 1 - S['PPV']
S['FOR'] = 1 - S['NPV']
S['ACC'] = S[['TP','TN']].sum(axis=1) / S[['TP','TN','FP','FN']].sum(axis=1)
S['W_ACC'] = 0.5 * S['TPR'] + 0.5 * S['TNR']
S['F1'] = 2 * (S['PPV'] * S['TPR']) / (S['PPV'] + S['TPR'])
print(S.drop(channels, axis=1))