import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

# Prevent log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MLP_NN(object):
	"""docstring for MLP_NN"""
	def __init__(self):
		super(MLP_NN, self).__init__()
		self.model = None
		# Reset previous states
		kr.backend.clear_session()

	def build(self, input_dim, layer, activation, optimizer, loss, metrics):
		self.model = kr.models.Sequential()
		self.model.add(kr.layers.Dense(layer[0], activation=activation[0], input_dim=input_dim))
		for i in range(1, len(layer)):
			self.model.add(kr.layers.Dense(layer[i], activation=activation[i]))
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	def train(self, X, y, val_size=None, epochs=10, verbose=0):
		history = None
		device_name = '/device:GPU:0' if tf.test.gpu_device_name() == '/device:GPU:0' else '/device:CPU:0'
		if verbose > 0:
			print('Training with {}'.format(device_name))
		with tf.device(device_name):
			history = self.model.fit(X, y, epochs=epochs, validation_split=val_size, shuffle=True, verbose=verbose)
		return history

	def prediction(self, X):
		pred = (self.model.predict(X) > 0.5).astype('int32').ravel()
		return pred

	def validation(self, D, wsize, wover, channels, label_id):
		# Lists to store the splited windows
		X_test = list()
		y_test = list()
		y_pred = list()
		# Step for window overlap
		wover_val = int(round(wover * wsize))
		step = wsize - wover_val
		# Index for window delimitation
		i, j = 0, wsize
		# Confusion matrix counters
		TP = FP = TN = FN = 0
		# Holder for posible False Negative
		possibleFN = False
		while j < D.index.size:
			win = D.iloc[i:j]
			real = 1 if 1 in win[label_id].values else 0
			x = win[channels].values.T.ravel()
			x = x.reshape(1, x.size)
			# Make the prediction
			pred = self.prediction(x)[0]
			# Store window and its labels
			X_test.append(win)
			y_test.append(real)
			y_pred.append(pred)
			# Confusion matrix
			if real == 1 and pred == 1:
				TP += 1
			elif real == 0 and pred == 1:
				FP += 1
			elif real == 0 and pred == 0:
				TN += 1
			elif real == 1 and pred == 0:
				possibleFN = True
			# Check possibles FN
			if real == 0 and possibleFN:
				FN += 1
				possibleFN = False
			# Next window index
			i += wsize if real == 1 and pred == 1 else step
			j += wsize if real == 1 and pred == 1 else step
		# Stats
		positives = TP + FN
		negatives = TN + FP
		accuracy = 0.0
		if positives != 0 and negatives != 0:
			accuracy = 0.5 * (TP / positives) + 0.5 * (TN / negatives)
		score = {
			'X_test': X_test,
			'y_test': y_test,
			'y_pred': y_pred,
			'tp': TP, 'fp': FP,
			'tn': TN, 'fn': FN,
			'accuracy': accuracy
		} 
		return score

	def save(self, filepath):
		kr.models.save_model(self.model, filepath)

	def load(self, filepath):
		self.model = kr.models.load_model(filepath)