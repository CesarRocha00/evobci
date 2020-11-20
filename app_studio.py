import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from eeg_utils import notch, bandpass
from sklearn.preprocessing import MinMaxScaler
from custom_widgets import FilePathDialog, VideoPlayer, CheckBoxPanel, FilteringForm, EEGViewer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStatusBar, QLineEdit, QStyle, QGroupBox, QGridLayout, QSlider, QLabel

class EEGStudio(QMainWindow):
	"""docstring for EEGStudio"""
	def __init__(self):
		super(EEGStudio, self).__init__()
		self.setWindowTitle('EEG Studio v1.0')
		self.setContentsMargins(10, 10, 10, 10)
		# Variables
		self.D = None
		self.X = None
		self.fs = 0
		self.label = list()
		self.markPos = list()
		self.markItem = list()
		self.maxDuration = 0
		# Main widgets
		self.fileDialog = FilePathDialog('EEG file:', 'Open file', 'EEG files (*.csv)')
		self.videoDialog = FilePathDialog('Video file:', 'Open video', 'Video files (*.mp4)')
		self.channelPanel = CheckBoxPanel(title='Channels')
		self.filteringForm = FilteringForm(title='Filtering')
		# Video player
		self.player = VideoPlayer()
		self.player.setSize(640, 480)
		self.player.error.connect(self.mediaError)
		self.player.stateChanged.connect(self.stateChanged)
		self.player.positionChanged.connect(self.positionChanged)
		self.player.durationChanged.connect(self.durationChanged)
		self.player.mediaStatusChanged.connect(self.mediaStatusChanged)
		self.player.videoAvailableChanged.connect(self.videoAvailableChanged)
		# Threads
		self.eegViewer = EEGViewer(mode='single')
		# EditLines
		self.timeEdit = QLineEdit('--:--:--')
		self.timeEdit.setAlignment(Qt.AlignCenter)
		self.indexEdit = QLineEdit('')
		self.indexEdit.setAlignment(Qt.AlignCenter)
		# Buttons
		self.icon = QWidget().style()
		self.loadButton = QPushButton('Load')
		self.loadButton.setIcon(self.icon.standardIcon(QStyle.SP_BrowserReload))
		self.loadButton.clicked.connect(self.loadFiles)
		self.saveButton = QPushButton('Save')
		self.saveButton.setIcon(self.icon.standardIcon(QStyle.SP_DialogSaveButton))
		self.saveButton.setEnabled(False)
		self.saveButton.clicked.connect(self.saveLabels)
		self.addButton = QPushButton('Add')
		self.addButton.setEnabled(False)
		self.addButton.clicked.connect(lambda: self.addLabel(None))
		self.applyButton = QPushButton('Apply')
		self.applyButton.setIcon(self.icon.standardIcon(QStyle.SP_DialogApplyButton))
		self.applyButton.setEnabled(False)
		self.applyButton.clicked.connect(self.applyParameters)
		self.playButton = QPushButton()
		self.playButton.setIcon(self.icon.standardIcon(QStyle.SP_MediaPlay))
		self.playButton.clicked.connect(self.play)
		self.playButton.setEnabled(False)
		self.stopButton = QPushButton()
		self.stopButton.setIcon(self.icon.standardIcon(QStyle.SP_MediaStop))
		self.stopButton.clicked.connect(self.stop)
		self.stopButton.setEnabled(False)
		self.findTimeButton = QPushButton('Time')
		self.findTimeButton.setEnabled(False)
		self.findTimeButton.clicked.connect(self.findTime)
		self.findIndexButton = QPushButton('Index')
		self.findIndexButton.setEnabled(False)
		self.findIndexButton.clicked.connect(self.findIndex)
		# Slider
		self.positionSlider = QSlider(Qt.Horizontal)
		self.positionSlider.setRange(0, 1)
		self.positionSlider.sliderMoved.connect(self.setPosition)
		# GroupBoxes
		searchGBox = QGroupBox('Search')
		searchGBox.setAlignment(Qt.AlignCenter)
		labelingGBox = QGroupBox('Labeling')
		labelingGBox.setAlignment(Qt.AlignCenter)
		# Layouts
		sourceLayout = QGridLayout()
		sourceLayout.setColumnStretch(0, 14)
		sourceLayout.setColumnStretch(1, 1)
		sourceLayout.setVerticalSpacing(0)
		sourceLayout.addWidget(self.fileDialog, 0, 0)
		sourceLayout.addWidget(self.videoDialog, 1, 0)
		sourceLayout.addWidget(self.loadButton, 0, 1, 2, 1)

		paramsLayout = QHBoxLayout()
		paramsLayout.addWidget(self.filteringForm, 7)
		paramsLayout.addWidget(self.applyButton,1)

		searchLayout = QGridLayout()
		searchLayout.addWidget(self.timeEdit, 0, 0, 1, 2, alignment=Qt.AlignCenter)
		searchLayout.addWidget(self.findTimeButton, 1, 0)
		searchLayout.addWidget(self.findIndexButton, 1, 1)
		searchGBox.setLayout(searchLayout)

		labelingLayout = QGridLayout()
		labelingLayout.addWidget(QLabel('Index:'), 0, 0, alignment=Qt.AlignRight)
		labelingLayout.addWidget(self.indexEdit, 0, 1)
		labelingLayout.addWidget(self.addButton, 0, 2)
		labelingLayout.addWidget(self.saveButton, 1, 0, 1, 3)
		labelingGBox.setLayout(labelingLayout)

		controlLayout = QHBoxLayout()
		controlLayout.addWidget(self.playButton, 1)
		controlLayout.addWidget(self.stopButton, 1)
		controlLayout.addWidget(self.positionSlider, 15)
		controlLayout.addWidget(searchGBox, 1)

		mainLayout = QVBoxLayout()
		mainLayout.addLayout(sourceLayout, 1)
		mainLayout.addStretch(1)
		mainLayout.addWidget(self.channelPanel, 1)
		mainLayout.addStretch(1)
		mainLayout.addLayout(paramsLayout, 1)
		mainLayout.addStretch(1)
		mainLayout.addWidget(labelingGBox, 1, alignment=Qt.AlignCenter)
		mainLayout.addStretch(1)
		mainLayout.addLayout(controlLayout, 1)
		# Main widget
		mainWidget = QWidget()
		mainWidget.setLayout(mainLayout)
		self.setCentralWidget(mainWidget)
		# Status bar
		self.statusBar = QStatusBar()
		self.setStatusBar(self.statusBar)
		self.statusBar.showMessage('¡System ready!')

	def loadFiles(self):
		try:
			path = self.fileDialog.getFullPath()
			self.D = pd.read_csv(path)
			self.statusBar.showMessage('EEG file was successfully loaded!')
			self.positionSlider.setRange(0, self.D.index.size)
			self.createChannelWidgets()
			self.eegViewer.show()
			path = self.videoDialog.getFullPath()
			self.player.loadMedia(path)
			self.applyButton.setEnabled(True)
		except FileNotFoundError:
			self.statusBar.showMessage('EEG file not found!')

	def mediaError(self):
		self.statusBar.showMessage('Error: {}'.format(self.player.errorString()))

	def setPosition(self, position):
		if self.player.isVideoAvailable():
			self.player.setPosition(position)
			self.findIndex()

	def durationChanged(self, duration):
		self.maxDuration = duration
		self.positionSlider.setRange(0, duration)
		self.timeEdit.setText('00:00:00')

	def positionChanged(self, position):
		self.positionSlider.setValue(position)
		s = position // 1000
		m = s // 60
		s = s % 60
		h = m // 60
		m = m % 60
		elapsed = '{:02d}:{:02d}:{:02d}'.format(h,m,s)
		self.timeEdit.setText(elapsed)

	def stateChanged(self, state):
		if state == VideoPlayer.PlayingState:
			self.playButton.setIcon(self.icon.standardIcon(QStyle.SP_MediaPause))
			self.statusBar.showMessage('Running...')
		else:
			self.playButton.setIcon(self.icon.standardIcon(QStyle.SP_MediaPlay))
			self.statusBar.showMessage('Paused!')
			
	def mediaStatusChanged(self, status):
		if status == self.player.LoadingMedia:
			self.statusBar.showMessage('Loading video...')
		elif status == self.player.LoadedMedia:
			self.player.stop()
			self.findTimeButton.setEnabled(True)
			self.findIndexButton.setEnabled(True)
			self.statusBar.showMessage('Video was successfully loaded!')

	def videoAvailableChanged(self, videoAvailable):
		if videoAvailable == True:
			self.player.show()
		else:
			self.player.hide()

	def createChannelWidgets(self):
		columns = self.D.columns
		self.channelPanel.setOptions(columns)

	def applyParameters(self):
		self.X = self.D.copy()
		param = self.filteringForm.getValues()
		channel = self.channelPanel.getChecked()
		color = self.channelPanel.getColors()
		numCh = param['n_channels']
		self.fs = param['fs']
		notch(self.X, self.fs, param['notch'], numCh)
		bandpass(self.X, self.fs, param['low'], param['high'], param['order'], numCh)
		self.X[self.X.columns[:numCh]] = MinMaxScaler().fit_transform(self.X[self.X.columns[:numCh]])
		self.label = np.zeros((self.X.index.size,), dtype=int)
		if 'Label' in self.X.columns:
			self.label = self.X['Label'].to_numpy(copy=True)
		self.eegViewer.configure(channel, color, self.fs * 1.2, self.fs)
		self.eegViewer.plotData(self.X)
		self.eegViewer.setPosition(0)
		self.markPos.clear()
		self.markItem.clear()
		marks = np.where(self.label == 1)[0]
		for index in marks:
			self.addLabel(index)
		self.addButton.setEnabled(True)
		self.playButton.setEnabled(True)
		self.playButton.setFocus()
		self.stopButton.setEnabled(True)
		self.saveButton.setEnabled(True)
		self.applyButton.setEnabled(False)

	def play(self):
		if self.player.isVideoAvailable():
			self.player.toggle()
		if self.eegViewer.getState() == EEGViewer.StoppedState:
			self.eegViewer.play()
		else:
			self.eegViewer.toggle()
		self.findIndex()
		self.playButton.setFocus()

	def stop(self):
		if self.player.isVideoAvailable():
			self.player.stop()
		self.eegViewer.stop()
		self.playButton.setFocus()
		self.applyButton.setEnabled(True)

	def timeToSeconds(self):
		text = self.timeEdit.text()
		h, m, s = text.split(':')
		seconds = int(h) * 3600 + int(m) * 60 + int(s)
		return seconds

	def findTime(self):
		position = self.timeToSeconds() * 1000
		if position <= self.maxDuration:
			self.player.setPosition(position)
		self.playButton.setFocus()

	def findIndex(self):
		position = self.timeToSeconds() * self.fs
		self.eegViewer.setPosition(position)
		self.playButton.setFocus()

	def addLabel(self, index=None):
		position = int(self.indexEdit.text().strip()) if index is None else index
		if position < 0 or position > self.D.index.size:
			self.statusBar.showMessage('Index {} does not exist!'.format(position))
		else:
			mark = self.eegViewer.addMark(position)
			mark.sigPositionChangeFinished.connect(self.markMoved)
			self.markPos.append(position)
			self.markItem.append(mark)
			self.label[position] = 1
			self.indexEdit.setText('')
			self.playButton.setFocus()

	def markMoved(self):
		markPos = np.array(self.markPos, dtype=int)
		linePos = np.array([line.getXPos() for line in self.markItem], dtype=int)
		moved = np.where(markPos != linePos)[0]
		old_index = markPos[moved]
		self.label[old_index] = 0
		new_index = linePos[moved]
		self.label[new_index] = 1
		self.markPos = linePos.tolist()
		self.statusBar.showMessage(f'Label moved from {old_index} to {new_index}')

	def saveLabels(self):
		self.D['Label'] = self.label
		path = self.fileDialog.getFullPath()
		path = path.split('.')[0]
		path += '_Label.csv'
		self.D.to_csv(path, index=False)
		self.statusBar.showMessage('Labeled EEG file saved as {}'.format(path))


import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	view = EEGStudio()
	view.show()
	sys.exit(app.exec_())