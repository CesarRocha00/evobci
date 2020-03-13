from scipy.signal import iirnotch, filtfilt, butter

# Function to split EEG (Pandas DataFrame) in secuential windows
def make_windows(D, wsize, labelID=None, wover=0):
	samples = D.index.size
	limit = samples - wsize
	step = wsize - wover
	X = [D.iloc[i:i + wsize] for i in range(0, limit, step)]
	y = [1 if 1 in win[labelID].values else 0 for win in X] if labelID is not None else []
	return X, y

def make_fixed_windows(D, wsize, labelID, wover=0, nwindows=1, step=1):
	half = wsize // 2
	remainder = wsize % 2
	padding = (nwindows // 2) * step
	include_ref = nwindows % 2 == 1
	X, y = list(), list()
	while True:
		targets = D[D[labelID] == 1].index
		if targets.size == 0:
			break
		ref = targets[0]
		# Compute limits
		start = ref - padding
		stop = ref + padding + 1
		# Window extraction
		for i in range(start, stop, step):
			if i == ref and not include_ref:
				continue
			left = i - half
			right = i + half + remainder
			X.append(D.iloc[left:right])
			y.append(1)
		# Remove central window
		left = ref - half
		right = ref + half + remainder
		D.drop(range(left, right), inplace=True)
		D.reset_index(drop=True, inplace=True)
	_X, _y = make_windows(D, wsize, labelID, wover)
	X.extend(_X)
	y.extend(_y)
	return X, y

# Compute de maximum allowed size for a EEG DataFrame segmentation
def max_window_size(D, labelID):
	label = D[labelID]
	index = label[label == 1].index
	# One minute in samples
	maxSize = 60 * 250
	for i in range(len(index) - 1):
		dist = index[i + 1] - index[i]
		if dist < maxSize: maxSize = dist
	return maxSize

# Refactor a dataset to have equal number of examples per class
def balance_dataset(X, y):
	pos = (y == 1).nonzero()[0]
	neg = (y == 0).nonzero()[0]
	tPos = pos.size
	fill = np.random.choice(neg, tPos, replace=False)
	index = np.concatenate((pos, fill), axis=0)
	return (X[index], y[index])

def notch(D, fs, w0, n_cols, Q=30.0):
	columns = D.columns[:n_cols]
	X = D[columns].values.T
	b, a = iirnotch(w0, Q, fs)
	X = filtfilt(b, a, X)
	D[columns] = X.T

def bandpass(D, fs, low, high, order, n_cols):
	columns = D.columns[:n_cols]
	X = D[columns].values.T
	nyq = fs * 0.5
	lowCut = low / nyq
	highCut = high / nyq
	b, a = butter(order, [lowCut, highCut], 'bandpass', output='ba')
	X = filtfilt(b, a, X)
	D[columns] = X.T