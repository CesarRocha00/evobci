from scipy.signal import iirnotch, filtfilt, butter

# Function to split EEG (Pandas DataFrame) in secuential windows
def make_windows(D, wsize, labelID=None, wover=0):
	samples = D.index.size
	limit = samples - wsize
	step = wsize - wover
	X = [D.iloc[i:i + wsize] for i in range(0, limit, step)]
	y = [1 if 1 in win[labelID].values else 0 for win in X] if labelID is not None else []
	return X, y

# Function to split EEG (Pandas DataFrame) in fixed windows with event
def make_fixed_windows(D, wsize, labelID, wover=0, step=1, padding=0):
	# Half and remainder of a windowx
	half = wsize // 2
	remainder = wsize % 2
	X, y = list(), list()
	# Re-sample each event in data
	while True:
		targets = D[D[labelID] == 1].index
		if targets.size == 0:
			break
		mark = targets[0]
		# Compute range bounds
		start = mark - half + padding
		stop = mark + half + remainder - padding + 1
		# Window extraction
		for i in range(start, stop, step):
			left = i - half
			right = i + half + remainder
			print(left, right)
			X.append(D.iloc[left:right])
			y.append(1)
		# Remove central window
		left = mark - half
		right = mark + half + remainder
		D.drop(range(left, right), inplace=True)
		D.reset_index(drop=True, inplace=True)
	_X, _y = make_windows(D, wsize, labelID, wover)
	X.extend(_X)
	y.extend(_y)

# Compute de maximum allowed size for a EEG DataFrame segmentation
def max_window_size(D, labelID):
	label = D[labelID]
	index = label[label == 1].index
	index = index.tolist()
	# Add left and right bounds
	index.insert(0, 0)
	index.append(D.index[-1])
	# One minute in samples
	maxSize = float('inf')
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

# Notch filter for EEG DataFrame
def notch(D, fs, w0, n_cols, Q=30.0):
	columns = D.columns[:n_cols]
	X = D[columns].values.T
	b, a = iirnotch(w0, Q, fs)
	X = filtfilt(b, a, X)
	D[columns] = X.T

# Bandpass filter for EEG DataFrame
def bandpass(D, fs, low, high, order, n_cols):
	columns = D.columns[:n_cols]
	X = D[columns].values.T
	nyq = fs * 0.5
	lowCut = low / nyq
	highCut = high / nyq
	b, a = butter(order, [lowCut, highCut], 'bandpass', output='ba')
	X = filtfilt(b, a, X)
	D[columns] = X.T