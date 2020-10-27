
from scipy.signal import iirnotch, filtfilt, butter

# Function to split EEG (Pandas DataFrame) into windows
def extract_windows(D, wsize, wover=0.0, label_id=None, fixed=False, padding=0.0, step=0.0):
	# Half and remainder of a windowx
	half = wsize // 2
	remainder = wsize % 2
	# Map percentages to samples
	wover_val = int(round(wover * wsize))
	padding_val = int(round(padding * wsize))
	# Check if resampling is required
	resampling = step > 0.0
	step_val = int(round(step * wsize)) if resampling else wsize
	# Lists to store the windows
	X, y = list(), list()
	_X, _y = list(), list()
	# Check for event (label) existence if fixed windows are required
	while fixed and 1 in D[label_id].values:
		# Get event index
		target = D[D[label_id] == 1].index[0]
		# Compute bounds for resampling
		start = target - half + padding_val if resampling else target
		stop = target + half + remainder + 1 - padding_val if resampling else target + 1
		# Slice dataframe into windows
		_X = [D.iloc[i - half:i + half + remainder] for i in range(start, stop, step_val)]
		X.extend(_X)
		y.extend([1] * len(_X))
		# Remove the central window
		left = target - half
		right = target + half + remainder
		D.drop(range(left, right), inplace=True)
		D.reset_index(drop=True, inplace=True)
	# Extract consecutive overlaped windows from remainder data
	samples = D.index.size
	stop = samples - wsize
	step_val = wsize - wover_val
	# Window extraction
	_X = [D.iloc[i:i + wsize] for i in range(0, stop, step_val)]
	if label_id is not None:
		_y = [1 if 1 in window[label_id].values else 0 for window in _X]
	X.extend(_X)
	y.extend(_y)
	# Return list of windows and classes
	return X, y

# Compute de maximum allowed size for a EEG DataFrame segmentation
def max_window_size(D, label_id):
	label = D[label_id]
	index = label[label == 1].index
	index = index.tolist()
	# Add left and right bounds
	index.insert(0, 0)
	index.append(D.index[-1])
	max_size = min([index[i + 1] - index[i] for i in range(len(index) - 1)])
	return max_size

# Refactor a dataset to have equal number of examples per class
def balance_dataset(X, y):
	pos = (y == 1).nonzero()[0]
	neg = (y == 0).nonzero()[0]
	total_pos = pos.size
	fill = np.random.choice(neg, total_pos, replace=False)
	index = np.concatenate((pos, fill), axis=0)
	return (X[index], y[index])

# Split an EEG based on a number (%) of events
def split_train_test(D, label_id, train_size):
	label = D[label_id]
	index = label[label == 1].index
	train_events = int(round(index.size * train_size))
	test_events = index.size - train_events
	a, b = index[train_events - 1], index[train_events]
	offset = (b - a) // 2
	cut_point = a + offset
	X_train = D.iloc[:cut_point]
	X_test = D.iloc[cut_point:]
	return (X_train, X_test)

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
	low_cut = low / nyq
	high_cut = high / nyq
	b, a = butter(order, [low_cut, high_cut], 'bandpass', output='ba')
	X = filtfilt(b, a, X)
	D[columns] = X.T