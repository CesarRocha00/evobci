import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from neuralnet import MLP_NN
from PyQt5.QtGui import QPixmap
from sklearn.preprocessing import MinMaxScaler
from eeg_utils import extract_windows, max_window_size, notch, bandpass
from custom_widgets import FilePathDialog, CheckBoxPanel, FilteringForm, RadioButtonPanel, EEGViewer
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QLabel, QPlainTextEdit, QCheckBox, QWidget, QStyle, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QStatusBar

class BCISimulator(QMainWindow):
	"""docstring for BCISimulator"""
	def __init__(self):
		super(BCISimulator, self).__init__()
		self.setWindowTitle('BCI Simulator v2.0')
		self.resize(1280, 720)
		self.setContentsMargins(10, 0, 10, 10)
		self.label_id = ''
		# Variables
		self.T = None
		self.V = None
		self.T_tmp = None
		self.V_tmp = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.y_pred = None
		self.param = dict()
		self.plotIndex = dict({'left': -1, 'right': -1})
		self.plotTotal = dict({'left': 0, 'right': 0})
		self.plotViewer = dict()
		self.pixmap = dict()
		self.indexLabel = dict()
		self.action = ''
		# Main widgets
		self.leftFileDialog = FilePathDialog('T. EEG file:', 'Open file', 'EEG files (*.csv)')
		self.rightFileDialog = FilePathDialog('V. EEG file:', 'Open file', 'EEG files (*.csv)')
		self.channelPanel = CheckBoxPanel(title='Channels')
		self.filteringForm = FilteringForm(title='Filtering')
		self.actionPanel = RadioButtonPanel(title='Action')
		self.actionPanel.setOptions(['Validation', 'Prediction'])
		# Threads
		self.plotViewer['left'] = EEGViewer(mode='single')
		self.plotViewer['right'] = EEGViewer(mode='single')
		# Pixmaps
		red = {'real': QPixmap('./icons/real-red.png'), 'pred': QPixmap('./icons/pred-red.png')}
		gray = {'real': QPixmap('./icons/real-gray.png'), 'pred': QPixmap('./icons/pred-gray.png')}
		green = {'real': QPixmap('./icons/real-green.png'), 'pred': QPixmap('./icons/pred-green.png')}
		self.pixmap = {'red': red, 'gray': gray, 'green': green}
		for c in self.pixmap.keys():
			for t in self.pixmap[c].keys():
				self.pixmap[c][t] = self.pixmap[c][t].scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		# Labels
		self.leftRealLabel = QLabel()
		self.leftRealLabel.setPixmap(self.pixmap['gray']['real'])
		self.rightRealLabel = QLabel()
		self.rightRealLabel.setPixmap(self.pixmap['gray']['real'])
		self.rightPredLabel = QLabel()
		self.rightPredLabel.setPixmap(self.pixmap['gray']['pred'])
		self.indexLabel['left'] = QLabel('0 / 0')
		self.indexLabel['right'] = QLabel('0 / 0')
		# EditLines
		self.wsizeEdit = QLineEdit('250')
		self.wsizeEdit.setAlignment(Qt.AlignCenter)
		self.woverEdit = QLineEdit('0.0')
		self.woverEdit.setAlignment(Qt.AlignCenter)
		self.wstepEdit = QLineEdit('0.0')
		self.wstepEdit.setAlignment(Qt.AlignCenter)
		self.wpaddEdit = QLineEdit('0.0')
		self.wpaddEdit.setAlignment(Qt.AlignCenter)
		self.epochEdit = QLineEdit('10')
		self.epochEdit.setAlignment(Qt.AlignCenter)
		self.statsEdit = QPlainTextEdit()
		# Checkboxes
		self.fixedCheck = QCheckBox('Fixed')
		self.fixedCheck.setChecked(False)
		# Buttons
		self.icon = QWidget().style()
		self.loadButton = QPushButton('Load')
		self.loadButton.setIcon(self.icon.standardIcon(QStyle.SP_BrowserReload))
		self.loadButton.clicked.connect(self.loadFiles)
		self.leftPrevButton = QPushButton()
		self.leftPrevButton.setIcon(self.icon.standardIcon(QStyle.SP_ArrowLeft))
		self.leftPrevButton.clicked.connect(lambda: self.updatePlot('left', -1))
		self.leftNextButton = QPushButton()
		self.leftNextButton.setIcon(self.icon.standardIcon(QStyle.SP_ArrowRight))
		self.leftNextButton.clicked.connect(lambda: self.updatePlot('left', 1))
		self.rightPrevButton = QPushButton()
		self.rightPrevButton.setIcon(self.icon.standardIcon(QStyle.SP_ArrowLeft))
		self.rightPrevButton.clicked.connect(lambda: self.updatePlot('right', -1))
		self.rightNextButton = QPushButton()
		self.rightNextButton.setIcon(self.icon.standardIcon(QStyle.SP_ArrowRight))
		self.rightNextButton.clicked.connect(lambda: self.updatePlot('right', 1))
		self.applyButton = QPushButton('Apply')
		self.applyButton.setIcon(self.icon.standardIcon(QStyle.SP_DialogApplyButton))
		self.applyButton.setEnabled(False)
		self.applyButton.clicked.connect(self.applyParameters)
		self.runButton = QPushButton('Run NN')
		self.runButton.setIcon(self.icon.standardIcon(QStyle.SP_MediaPlay))
		self.runButton.clicked.connect(self.runExperiment)
		self.runButton.setEnabled(False)
		# GroupBoxes
		windowGBox = QGroupBox('Window')
		windowGBox.setAlignment(Qt.AlignCenter)
		trainingGBox = QGroupBox('Training')
		trainingGBox.setAlignment(Qt.AlignCenter)
		# Layouts
		leftInfoLayout = QHBoxLayout()
		leftInfoLayout.addWidget(self.leftRealLabel, alignment=Qt.AlignLeft)
		leftInfoLayout.addStretch(1)
		leftInfoLayout.addWidget(self.leftPrevButton, alignment=Qt.AlignCenter)
		leftInfoLayout.addWidget(self.leftNextButton, alignment=Qt.AlignCenter)
		leftInfoLayout.addStretch(1)
		leftInfoLayout.addWidget(self.indexLabel['left'], alignment=Qt.AlignRight)

		leftViewLayout = QVBoxLayout()
		leftViewLayout.addWidget(self.leftFileDialog)
		leftViewLayout.addWidget(self.plotViewer['left'].widget())
		leftViewLayout.addLayout(leftInfoLayout)

		middleViewLayout = QVBoxLayout()
		middleViewLayout.addStretch(6)
		middleViewLayout.addWidget(self.loadButton, alignment=Qt.AlignCenter)
		middleViewLayout.addStretch(1)
		middleViewLayout.addWidget(self.statsEdit)
		middleViewLayout.addStretch(6)

		rightInfoLayout = QHBoxLayout()
		rightInfoLayout.addWidget(self.rightRealLabel, alignment=Qt.AlignLeft)
		rightInfoLayout.addWidget(self.rightPredLabel, alignment=Qt.AlignLeft)
		rightInfoLayout.addStretch(1)
		rightInfoLayout.addWidget(self.rightPrevButton, alignment=Qt.AlignCenter)
		rightInfoLayout.addWidget(self.rightNextButton, alignment=Qt.AlignCenter)
		rightInfoLayout.addStretch(1)
		rightInfoLayout.addWidget(self.indexLabel['right'], alignment=Qt.AlignRight)

		rightViewLayout = QVBoxLayout()
		rightViewLayout.addWidget(self.rightFileDialog)
		rightViewLayout.addWidget(self.plotViewer['right'].widget())
		rightViewLayout.addLayout(rightInfoLayout)

		viewLayout = QHBoxLayout()
		viewLayout.addLayout(leftViewLayout, 6)
		viewLayout.addLayout(middleViewLayout, 2)
		viewLayout.addLayout(rightViewLayout, 6)

		windowLayout = QGridLayout()
		windowLayout.addWidget(QLabel('Size (samples)'), 0, 0, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wsizeEdit, 1, 0, alignment=Qt.AlignCenter)
		windowLayout.addWidget(QLabel('Overlap (%)'), 0, 1, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.woverEdit, 1, 1, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.fixedCheck, 0, 2, 2, 1, alignment=Qt.AlignCenter)
		windowLayout.addWidget(QLabel('Padding (%)'), 0, 3, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wpaddEdit, 1, 3, alignment=Qt.AlignCenter)
		windowLayout.addWidget(QLabel('Step (%)'), 0, 4, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wstepEdit, 1, 4, alignment=Qt.AlignCenter)
		windowGBox.setLayout(windowLayout)

		trainingLayout = QGridLayout()
		trainingLayout.addWidget(QLabel('Epochs'), 0, 0, alignment=Qt.AlignCenter)
		trainingLayout.addWidget(self.epochEdit, 1, 0, alignment=Qt.AlignCenter)
		trainingGBox.setLayout(trainingLayout)

		paramsLayout = QHBoxLayout()
		paramsLayout.addWidget(self.filteringForm, 6)
		paramsLayout.addWidget(windowGBox, 5)
		paramsLayout.addWidget(trainingGBox, 1)

		actionLayout = QHBoxLayout()
		actionLayout.addStretch()
		actionLayout.addWidget(self.actionPanel, alignment=Qt.AlignCenter)
		actionLayout.addWidget(self.applyButton, alignment=Qt.AlignCenter)
		actionLayout.addStretch()

		mainLayout = QVBoxLayout()
		mainLayout.addLayout(viewLayout)
		mainLayout.addWidget(self.channelPanel)
		mainLayout.addLayout(paramsLayout)
		mainLayout.addLayout(actionLayout)
		mainLayout.addWidget(self.runButton, alignment=Qt.AlignCenter)
		# Main widget
		mainWidget = QWidget()
		mainWidget.setLayout(mainLayout)
		self.setCentralWidget(mainWidget)
		# Status bar
		self.statusBar = QStatusBar()
		self.setStatusBar(self.statusBar)
		self.statusBar.showMessage('¡System ready!')

	def loadFiles(self):
		# Training file
		path = self.leftFileDialog.getFullPath()
		try:
			self.T = pd.read_csv(path)
			self.statusBar.showMessage('Training EEG file was successfully loaded!')
			self.createChannelWidgets()
		except:
			self.T = None
			self.statusBar.showMessage('Training EEG file not found!')
		# Validation file
		path = self.rightFileDialog.getFullPath()
		try:
			self.V = pd.read_csv(path)
			self.statusBar.showMessage('Validation EEG file was successfully loaded!')
		except:
			self.V = None
			self.statusBar.showMessage('Validation EEG file not found!')
		# Enable Apply button
		self.applyButton.setEnabled(True)

	def createChannelWidgets(self):
		self.label_id = self.T.columns[-1]
		column_names = self.T.columns[:-1]
		self.channelPanel.setOptions(column_names)

	def applyParameters(self):
		selected_channels = self.channelPanel.getChecked()
		if len(selected_channels) == 0:
			self.statusBar.showMessage('You must select at least one channel!')
			return
		# Data backup and preprocesing
		filters = self.filteringForm.getValues()
		# Training
		if self.T is not None:
			self.T_tmp = self.applyFilters(self.T, filters, selected_channels)
		else:
			self.statusBar.showMessage('Training EEG file was not loaded correctly! Try again.')
			return
		# Validation
		if self.V is not None:
			self.V_tmp = self.applyFilters(self.V, filters, selected_channels)
		# Window parameters
		self.param['wsize'] = int(self.wsizeEdit.text().strip())
		self.param['wover'] = float(self.woverEdit.text().strip())
		self.param['fixed'] = self.fixedCheck.isChecked()
		self.param['wpadd'] = float(self.wpaddEdit.text().strip())
		self.param['wstep'] = float(self.wstepEdit.text().strip())
		self.param['label'] = self.label_id
		self.param['channels'] = selected_channels
		# Training parameters
		self.param['epochs'] = int(self.epochEdit.text().strip())
		# Reset labels and counters
		self.resetStates()
		# Now try to create the traininig and testing datasets
		self.createDatasets(selected_channels)

	def applyFilters(self, D, filters, channels):
		C = D.copy()
		notch(C, filters['fs'], filters['notch'], filters['n_channels'])
		bandpass(C, filters['fs'], filters['low'], filters['high'], filters['order'], filters['n_channels'])
		C[channels] = MinMaxScaler().fit_transform(C[channels])
		# Remove non selected channels
		if self.label_id in C.columns:	
			channels.append(self.label_id)
			C = C[channels]
			channels.pop()
		else:
			C = C[channels]
		return C

	def createDatasets(self, channels):
		colors = self.channelPanel.getColors()
		self.action = self.actionPanel.getChecked()
		# Verify if fixed windows will be used
		maxWsize = max_window_size(self.T_tmp, self.label_id)
		if self.param['fixed'] and self.param['wsize'] > maxWsize:
			self.statusBar.showMessage('There are some fixed windows overlaped. Check the window size.')
		# Split training data into windows
		self.T_tmp, self.y_train = extract_windows(self.T_tmp, self.param['wsize'], self.param['wover'], self.label_id, self.param['fixed'], self.param['wpadd'], self.param['wstep'])
		# Map lists of windows to numpy arrays
		self.X_train = np.array([win[channels].values.T.ravel() for win in self.T_tmp])
		self.y_train = np.array(self.y_train)
		# Check validation type to create testing dataset 
		if self.action == 'Prediction' and self.V is not None:
			# Split validation data into windows
			self.V_tmp, self.y_test = extract_windows(self.V_tmp, self.param['wsize'], self.param['wover'], None, False, 0.0, 0.0)
			# Map lists of windows to numpy arrays
			self.X_test = np.array([win[channels].values.T.ravel() for win in self.V_tmp])
			self.y_test = np.array(self.y_test)
		else:
			# Add validation for label_id existence in Validation file
			pass
		# Message of dataset creation
		self.statusBar.showMessage('Datasets are ready!')
		# EEG viewer setup
		self.plotViewer['left'].configure(channels, colors, self.param['wsize'])
		self.plotViewer['right'].configure(channels, colors, self.param['wsize'])
		# Set plot counter
		self.plotTotal['left'] = len(self.T_tmp)
		# Plot first training window
		self.updatePlot('left', 1)
		# Activate run button
		self.runButton.setEnabled(True)

	def runExperiment(self):
		model = MLP_NN()
		# Input and output dimension
		input_dim = self.X_train[0].size
		output_dim = 1
		# Two hidden layers
		ratio_io = int((input_dim / output_dim) ** (1 / 3))
		hidden_1 = output_dim * (ratio_io ** 2)
		hidden_2 = output_dim * ratio_io
		layer = [hidden_1, hidden_2, output_dim]
		activation = ['relu', 'relu', 'sigmoid']
		# Optimizer and metrics
		optimizer = 'adam'
		loss = 'mse'
		metrics = ['accuracy']
		self.statusBar.showMessage('Building neural network...')
		# Configure neural network
		model.build(input_dim, layer, activation, optimizer, loss, metrics)
		self.statusBar.showMessage('Running training phase...')
		# Perform the training
		model.train(self.X_train, self.y_train, val_size=0.3, epochs=self.param['epochs'], verbose=0)
		# Get results
		if self.action == 'Prediction':
			score = model.prediction(self.X_test)
			self.y_pred = score
			self.updateStats(score, 'prediction')
		else:
			score = model.validation(self.V_tmp, self.param['wsize'], self.param['wover'], self.param['channels'], self.label_id)
			self.V_tmp, self.y_test, self.y_pred = score['X_test'], score['y_test'], score['y_pred']
			self.updateStats(score, 'validation')
		self.statusBar.showMessage('Neural network has finished!')
		# Set plot counter
		self.plotTotal['right'] = len(self.V_tmp)
		# Plot first testing window
		self.updatePlot('right', 1)
		# Deactivate run button
		self.runButton.setEnabled(False)

	def updatePlot(self, side, val):
		if (self.plotIndex[side] + val) >= 0 and (self.plotIndex[side] + val) < self.plotTotal[side]:
			self.plotIndex[side] += val
			data = self.T_tmp[self.plotIndex[side]] if side == 'left' else self.V_tmp[self.plotIndex[side]]
			self.plotViewer[side].plotData(data)
			self.plotViewer[side].update()
			self.updateLabel(side)

	def updateLabel(self, side):
		color = {0: 'red', 1: 'green'}
		if side == 'left':
			pixReal = self.pixmap[color[self.y_train[self.plotIndex[side]]]]['real']
			self.leftRealLabel.setPixmap(pixReal)
		else:
			pixReal = self.pixmap['gray']['real']
			if len(self.y_test) > 0:
				pixReal = self.pixmap[color[self.y_test[self.plotIndex[side]]]]['real']
			pixPred = self.pixmap[color[self.y_pred[self.plotIndex[side]]]]['pred']
			self.rightRealLabel.setPixmap(pixReal)
			self.rightPredLabel.setPixmap(pixPred)
		# Upadate index label
		self.indexLabel[side].setText('{} / {}'.format(self.plotIndex[side] + 1, self.plotTotal[side]))

	def updateStats(self, score, desc):
		if desc == 'validation':
			summary = 'VALIDATION\n'
			summary += '-> TP: {}\n-> FP: {}\n'.format(score['tp'], score['fp'])
			summary += '-> TN: {}\n-> FN: {}\n'.format(score['tn'], score['fn'])
			summary += '-> Accuracy: {}\n'.format(round(score['accuracy'], 5))
			self.statsEdit.insertPlainText(summary)
		else:
			positives = score.sum()
			negatives = score.size - positives
			summary = 'PREDICTION\n'
			summary += '-> Positives: {}\n-> Negatives: {}\n'.format(positives, negatives)
			self.statsEdit.insertPlainText(summary)

	def resetStates(self):
		self.plotIndex['left'] = -1
		self.plotIndex['right'] = -1
		self.plotTotal['left'] = 0
		self.plotTotal['right'] = 0
		self.statsEdit.clear()
		self.indexLabel['left'].setText('0 / 0')
		self.indexLabel['right'].setText('0 / 0')
		self.leftRealLabel.setPixmap(self.pixmap['gray']['real'])
		self.rightRealLabel.setPixmap(self.pixmap['gray']['real'])
		self.rightPredLabel.setPixmap(self.pixmap['gray']['pred'])