import sys
import pandas as pd
from datetime import timedelta

d  = pd.read_csv(sys.argv[1])
seconds = d['elapsed'].sum()
print(f'{str(timedelta(seconds=round(seconds))):0>8}')