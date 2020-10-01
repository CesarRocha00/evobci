import os
# Silence every warning of notice from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras as kr

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
		is_gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
		device_name = tf.test.gpu_device_name() if is_gpu_available else '/device:CPU:0'
		with tf.device(device_name):
			history = self.model.fit(X, y, epochs=epochs, validation_split=val_size, shuffle=True, verbose=verbose)
		return history

	def predict(self, X):
		pred = (self.model.predict(X) > 0.5).astype('int32').ravel()
		return pred

	def save(self, filepath):
		kr.models.save_model(self.model, filepath)

	def load(self, filepath):
		self.model = kr.models.load_model(filepath)