import cv2
import h5py
import numpy as np
import pandas as pd
from math import ceil
from time import sleep
from datetime import datetime
# EEG Tools
from eegtools import *
# Artificial Neural Network
from neuralnet import MLP_NN
# Lab Streaming Layer
from pylsl import StreamInlet, resolve_stream
# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# PyQtGraph
from pyqtgraph import mkPen, GraphicsLayoutWidget
# PyQt5
from PyQt5.QtGui import QPixmap,  QIcon
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import (Qt, QDir, QUrl, QFile, QThread)
from PyQt5.QtWidgets import (QMainWindow, QLabel, QStatusBar, QVBoxLayout, QHBoxLayout,
							 QPushButton, QWidget, QGroupBox, QLineEdit, QSizePolicy,
							 QSlider, QStyle, QFileDialog, QRadioButton, QFormLayout, 
							 QGridLayout, QDesktopWidget, QCheckBox, QPlainTextEdit)

__version__ = '0.1'
__author__ = 'César Rocha'

class FilePathDialog(QWidget):
	"""docstring for FilePathDialog"""
	def __init__(self, label, caption, filters):
		super(FilePathDialog, self).__init__()
		self.caption = caption
		self.filters = filters
		self.fullPath = None
		layout = QHBoxLayout()
		self.setLayout(layout)
		self.pathEdit = QLineEdit()
		self.pathEdit.setReadOnly(True)
		openButton = QPushButton()
		openButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
		openButton.clicked.connect(self.openFile)
		layout.addWidget(QLabel(label))
		layout.addWidget(self.pathEdit)
		layout.addWidget(openButton)

	def openFile(self):
		self.fullPath, _ = QFileDialog.getOpenFileName(self, self.caption, QDir.homePath(), self.filters)
		self.pathEdit.setText(self.fullPath)

	def getFullPath(self):
		return self.fullPath


class LSLForm(QGroupBox):
	"""docstring for LSLForm"""

	fields = ['server', 'channels']
	labels = ['Server name:', 'Channels (CSV):']

	def __init__(self, title=None, flat=False):
		super(LSLForm, self).__init__()
		if title is not None:
			self.setTitle(title)
		self.setFlat(flat)
		self.setAlignment(Qt.AlignCenter)
		layout = QFormLayout()
		layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
		self.setLayout(layout)
		self.control = dict()
		for i, field in enumerate(self.fields):
			edit = QLineEdit()
			layout.addRow(QLabel(self.labels[i]), edit)
			self.control[field] = edit

	def getValues(self):
		return {field: self.control[field].text().strip() for field in self.fields}


class FilteringForm(QGroupBox):
	"""docstring for FilteringForm"""

	content = ['250','60','1','12','5','8']
	fields = ['fs','notch','low','high','order','channels']
	labels = ['Fs (Hz)','Notch (Hz)','Low Cut (Hz)','Hight Cut (Hz)','Order', 'Num. Channels']

	def __init__(self, title=None, flat=False):
		super(FilteringForm, self).__init__()
		if title is not None:
			self.setTitle(title)
		self.setFlat(flat)
		self.setAlignment(Qt.AlignCenter)
		layout = QGridLayout()
		self.setLayout(layout)
		self.control = dict()
		for i, field in enumerate(self.fields):
			edit = QLineEdit(self.content[i])
			edit.setAlignment(Qt.AlignCenter)
			layout.addWidget(QLabel(self.labels[i]), 0, i, alignment=Qt.AlignCenter)
			layout.addWidget(edit, 1, i, alignment=Qt.AlignCenter)
			self.control[field] = edit

	def getValues(self):
		return {field: int(self.control[field].text().strip()) for field in self.fields}


class PersonalInformationForm(QGroupBox):
	"""docstring for PersonalInformationForm"""

	fields = ['name','last','age','sex']
	labels = ['First name:','Last name:','Age:','Sex:']

	def __init__(self, title=None, flat=False):
		super(PersonalInformationForm, self).__init__()
		if title is not None:
			self.setTitle(title)
		self.setFlat(flat)
		self.setAlignment(Qt.AlignCenter)
		layout = QFormLayout()
		layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
		self.setLayout(layout)
		self.control = dict()
		for i, field in enumerate(self.fields[:-1]):
			edit = QLineEdit()
			layout.addRow(QLabel(self.labels[i]), edit)
			self.control[field] = edit
		# Add sex radio button panel
		panel = RadioButtonPanel(flat=True)
		panel.setOptions(['Male', 'Female'])
		layout.addRow(QLabel(self.labels[-1]), panel)
		self.control[self.fields[-1]] = panel

	def getValues(self):
		values = {field: self.control[field].text().strip() for field in self.fields[:-1]}
		values[self.fields[-1]] = self.control[self.fields[-1]].getChecked()
		return values


class CheckBoxPanel(QGroupBox):
	"""docstring for CheckBoxPanel"""
	def __init__(self, title=None, flat=False, columns=10):
		super(CheckBoxPanel, self).__init__()
		if title is not None:
			self.setTitle(title)
		self.setFlat(flat)
		self.setAlignment(Qt.AlignCenter)
		self.columns = columns
		self.layout = QGridLayout()
		self.layout.setVerticalSpacing(20)
		self.setLayout(self.layout)
		self.color = dict()
		self.control = list()
		self.checked = list()

	def setOptions(self, names):
		self.reset()
		# Create and add new controls
		row = 0
		colLimit = self.columns
		for i, label in enumerate(names):
			self.color[label] = '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))
			option = QCheckBox(label)
			option.setStyleSheet('background: {}'.format(self.color[label]))
			option.stateChanged.connect(self.newChange)
			self.layout.addWidget(option, row, i % colLimit, alignment=Qt.AlignCenter)
			self.control.append(option)
			if (i + 1) % colLimit == 0:
				row += 1

	def newChange(self):
		self.checked = [check.text() for check in self.control if check.isChecked()]

	def getColors(self):
		return self.color

	def getChecked(self):
		return self.checked

	def reset(self):
		# Remove previous checkboxes
		for i in reversed(range(self.layout.count())):
			check = self.layout.itemAt(i).widget()
			self.layout.removeWidget(check)
			check.deleteLater()
		# Clear control list and color dictionary
		self.color.clear()
		self.control.clear()


class RadioButtonPanel(QGroupBox):
	"""docstring for RadioButtonPanel"""
	def __init__(self, title=None, flat=False):
		super(RadioButtonPanel, self).__init__()
		if title is not None:
			self.setTitle(title)
		self.setFlat(flat)
		self.setAlignment(Qt.AlignCenter)
		self.layout = QHBoxLayout()
		self.setLayout(self.layout)
		self.control = list()
		self.checked = None

	def setOptions(self, names):
		self.control.clear()
		for label in names:
			option = QRadioButton(label)
			option.toggled.connect(self.newChange)
			self.layout.addWidget(option, alignment=Qt.AlignCenter)
			self.control.append(option)
		if len(names) > 0:
			self.control[0].setChecked(True)

	def newChange(self):
		self.checked = [radio.text() for radio in self.control if radio.isChecked()][0]

	def getChecked(self):
		return self.checked


class VideoPlayer(QMediaPlayer):
	"""docstring for VideoPlayer"""
	def __init__(self):
		super(VideoPlayer, self).__init__(None, QMediaPlayer.VideoSurface)
		self.view = QVideoWidget()
		self.view.setWindowTitle('Video Player')
		self.setVideoOutput(self.view)

	def widget(self):
		return self.view

	def show(self):
		self.view.show()

	def hide(self):
		self.view.hide()

	def isVisible(self):
		return self.view.isVisible()

	def setSize(self, width, height):
		self.view.resize(width, height)

	def toggle(self):
		if self.state() == self.PlayingState:
			self.pause()
		else:
			self.play()

	def loadMedia(self, fullPath):
		self.setMedia(QMediaContent(QUrl.fromLocalFile(fullPath)))
		self.stop()


class EEGViewer(QThread):
	"""docstring for EEGViewer"""
	
	StoppedState = 0
	PausedState = 1
	RunningState = 2

	def __init__(self, mode='single', rows=4):
		super(EEGViewer, self).__init__()
		self.mode = mode
		self.rows = rows
		self.view = GraphicsLayoutWidget()
		self.view.setAntialiasing(True)
		self.view.setWindowTitle('EEG Viewer')
		self.state = self.StoppedState
		self.position = 0
		self.maxPosition = 0
		self.plotItem = list()
		self.plotTrace = dict()
		# Holders
		self.wait = 0
		self.wsize = 0
		self.hsize = 0
		self.color = dict()
		self.window = list([0, 0])
		self.channel = list()

	def widget(self):
		return self.view

	def show(self):
		self.view.show()

	def hide(self):
		self.view.hide()

	def getState(self):
		return self.state

	def isVisible(self):
		return self.view.isVisible()

	def setSize(self, width, height):
		self.view.resize(width, height)

	def configure(self, channel, color, wsize, fs=0):
		# Link params
		nCh = len(channel)
		self.wait = 1 / (fs * nCh) if fs > 0 else 0
		self.wsize = wsize
		self.hsize = wsize / 2
		self.color = color
		self.channel = channel
		self.window = np.array([0, wsize])
		# Remove previous items and traces
		self.view.clear()
		self.plotItem.clear()
		self.plotTrace.clear()
		# Create new canvas
		if self.mode == 'single':
			self.singleLayout()
		else:
			self.multipleLayout()

	def singleLayout(self):
		canvas = self.view.addPlot(0, 0)
		canvas.disableAutoRange()
		canvas.setClipToView(True)
		canvas.setLimits(yMin=0, yMax=1)
		canvas.setDownsampling(mode='subsample')
		canvas.showGrid(x=True, y=True, alpha=0.25)
		for ch in self.channel:
			pen = mkPen(color=self.color[ch], width=2)
			self.plotTrace[ch] = canvas.plot(pen=pen)
		self.plotItem.append(canvas)

	def multipleLayout(self):
		col = 0
		rowLimit = self.rows
		for i, ch in enumerate(self.channel):
			pen = mkPen(color=self.color[ch], width=2)
			canvas = self.view.addPlot(i % rowLimit, col)
			canvas.disableAutoRange()
			canvas.setClipToView(True)
			canvas.setLimits(yMin=0, yMax=1)
			canvas.setDownsampling(mode='subsample')
			canvas.showGrid(x=True, y=True, alpha=0.25)
			self.plotItem.append(canvas)
			self.plotTrace[ch] = canvas.plot(pen=pen)
			if (i + 1) % rowLimit == 0:
				col += 1

	def plotData(self, D):
		for ch in self.channel:
			self.plotTrace[ch].setData(D[ch].values)
		self.position = 0
		self.maxPosition = D.index.size

	def addMark(self, position):
		for canvas in self.plotItem:
			canvas.addLine(x=position)

	def setPosition(self, position):
		self.window[0] = position - self.hsize
		self.window[1] = position + self.hsize
		self.position = position
		self.update()

	def update(self):
		for plot in self.plotItem:
			plot.setRange(xRange=self.window)
		self.position += 1 if self.position < self.maxPosition else 0

	def play(self):
		self.state = self.RunningState
		self.start()

	def pause(self):
		self.state = self.PausedState

	def toggle(self):
		self.state = self.PausedState if self.state == self.RunningState else self.RunningState

	def stop(self):
		self.state = self.StoppedState
		self.quit()
		self.setPosition(0)

	def run(self):
		while True:
			if self.state == self.RunningState:
				self.setPosition(self.position)
			elif self.state == self.PausedState:
				pass
			else:
				break
			sleep(self.wait)


class OCVWebcam(QThread):
	"""docstring for OCVWebcam"""

	StoppedState = 0
	PausedState = 1
	RecordingState = 2

	def __init__(self):
		super(OCVWebcam, self).__init__()
		self.source = None
		self.state = self.StoppedState
		self.file = None
		self.width = 0
		self.height = 0
		self.elapsed = 0
		self.time_counter = 0
		self.frame_counter = 0

	def setSource(self, src, width=640, height=480):
		self.source = cv2.VideoCapture(src)
		self.width = width
		self.height = height
		self.source.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	def updateElapsedTime(self):
		difference = datetime.now() - self.elapsed
		self.time_counter += difference.total_seconds()

	def record(self, filename):
		self.file = h5py.File(filename, 'w')
		self.state = self.RecordingState
		self.time_counter = 0
		self.frame_counter = 0
		self.elapsed = datetime.now()
		self.start()

	def pause(self):
		self.state = self.PausedState
		self.updateElapsedTime()

	def toggle(self):
		self.state = self.PausedState if self.state == self.RecordingState else self.RecordingState
		if self.state == self.PausedState:
			self.updateElapsedTime()
		else:
			self.elapsed = datetime.now()

	def stop(self):
		self.state = self.StoppedState
		self.quit()
		while not self.isFinished():
			pass
		self.updateElapsedTime()
		# Add attributes to file
		fps = self.frame_counter / self.time_counter
		self.file.attrs['fps'] = fps
		self.file.attrs['width'] = self.width
		self.file.attrs['height'] = self.height
		self.file.attrs['length'] = self.frame_counter
		self.file.attrs['duration'] = self.time_counter
		self.file.close()
		self.source.release()

	def run(self):
		while True:
			if self.state == self.RecordingState:
				ret, frame = self.source.read()
				self.file.create_dataset('{}'.format(self.frame_counter), data=frame)
				self.frame_counter += 1
			elif self.state == self.PausedState:
				pass
			else:
				break


class LSLClient(QThread):
	"""docstring for LSLClient"""

	StoppedState = 0
	PausedState = 1
	RecordingState = 2

	def __init__(self):
		super(LSLClient, self).__init__()
		self.D = list()
		self.T = list()
		self.state = self.StoppedState
		self.streams = None
		self.inlet = None

	def setStream(self, tag, value):
		self.streams = resolve_stream(tag, value)
		self.inlet = StreamInlet(self.streams[0])
		self.D.clear()
		self.T.clear()

	def getData(self):
		return self.D

	def getState(self):
		return self.state

	def record(self):
		self.state = self.RecordingState
		self.start()

	def pause(self):
		self.state = self.PausedState

	def toggle(self):
		self.state = self.PausedState if self.state == self.RecordingState else self.RecordingState

	def stop(self):
		self.state = self.StoppedState
		self.quit()

	def run(self):
		while True:
			if self.state == self.RecordingState:
				chunk, timestamps = self.inlet.pull_chunk()
				self.D.extend(chunk)
			elif self.state == self.PausedState:
				pass
			else:
				break


class EEGRecorder(QMainWindow):
	"""docstring for EEGRecorder"""
	def __init__(self):
		super(EEGRecorder, self).__init__()
		self.setWindowTitle('EEG Recorder v2.0')
		self.setWindowIcon(QIcon('./icons/record.png'))
		self.resize(480,640)
		self.setContentsMargins(10, 10, 10, 10)
		# Main widgets
		self.videoDialog = FilePathDialog('Video file:', 'Open video', 'Video files (*.mp4)')
		self.connectionForm = LSLForm('Lab Streaming Layer')
		self.personalForm = PersonalInformationForm('Personal Information')
		# Video player
		self.player = VideoPlayer()
		self.player.setSize(640, 480)
		self.player.error.connect(self.mediaError)
		self.player.mediaStatusChanged.connect(self.mediaStatusChanged)
		# Threads
		self.lsl = LSLClient()
		self.camera = OCVWebcam()
		# Buttons
		self.loadButton = QPushButton('Load')
		self.loadButton.clicked.connect(self.loadVideo)
		self.startButton = QPushButton('Start')
		self.startButton.setEnabled(False)
		self.startButton.clicked.connect(self.startRecording)
		self.pauseResumeButton = QPushButton('Pause')
		self.pauseResumeButton.setEnabled(False)
		self.pauseResumeButton.clicked.connect(self.pauseAndResume)
		self.stopSaveButton = QPushButton('Stop && Save')
		self.stopSaveButton.setEnabled(False)
		self.stopSaveButton.clicked.connect(self.stopAndSave)
		# Layouts
		buttonLayout = QHBoxLayout()
		buttonLayout.addWidget(self.startButton)
		buttonLayout.addWidget(self.pauseResumeButton)
		buttonLayout.addWidget(self.stopSaveButton)
		mainLayout = QVBoxLayout()
		mainLayout.addWidget(self.videoDialog, 1)
		mainLayout.addStretch(1)
		mainLayout.addWidget(self.loadButton)
		mainLayout.addStretch(1)
		mainLayout.addWidget(self.connectionForm, 1)
		mainLayout.addStretch(1)
		mainLayout.addWidget(self.personalForm, 1)
		mainLayout.addStretch(1)
		mainLayout.addLayout(buttonLayout, 1)
		# Main widget
		mainWidget = QWidget()
		mainWidget.setLayout(mainLayout)
		self.setCentralWidget(mainWidget)
		# Status bar
		self.statusBar = QStatusBar()
		self.setStatusBar(self.statusBar)
		self.statusBar.showMessage('¡System ready!')

	def loadVideo(self):
		path = self.videoDialog.getFullPath()
		self.player.loadMedia(path)
		self.camera.setSource(0, width=640, height=480)
		self.startButton.setEnabled(True)

	def mediaError(self):
		self.statusBar.showMessage('Error: {}'.format(self.player.errorString()))

	def mediaStatusChanged(self, status):
		if status == self.player.LoadingMedia:
			self.statusBar.showMessage('Loading video...')
		elif status == self.player.LoadedMedia:
			self.player.show()
			self.statusBar.showMessage('Video was successfully loaded!')
		elif status == self.player.EndOfMedia:
			self.stopAndSave()

	def startRecording(self):
		server = self.connectionForm.getValues()['server']
		if server == '':
			self.statusBar.showMessage('¡LSL server not specified!')
			return
		self.lsl.setStream('name', server)
		self.camera.record('video.hdf5')
		self.lsl.record()
		if self.player.isVideoAvailable():
			self.player.play()
		self.statusBar.showMessage('Recording...')
		self.pauseResumeButton.setEnabled(True)
		self.stopSaveButton.setEnabled(True)
		self.startButton.setEnabled(False)
		self.loadButton.setEnabled(False)

	def pauseAndResume(self):
		self.lsl.toggle()
		self.camera.toggle()
		if self.player.isVideoAvailable():
			self.player.toggle()
		if self.lsl.getState() == LSLClient.PausedState:
			self.pauseResumeButton.setText('Resume')
			self.statusBar.showMessage('Paused!')
		else:
			self.pauseResumeButton.setText('Pause')
			self.statusBar.showMessage('Recording...')

	def stopAndSave(self):
		self.lsl.stop()
		self.camera.stop()
		self.statusBar.showMessage('Stopping LSL thread...')
		while not self.lsl.isFinished():
			pass
		self.statusBar.showMessage('LSL thread finished!')
		if self.player.isVideoAvailable():
			self.player.stop()
			self.player.hide()
		path = QFileDialog.getExistingDirectory(self, 'Select folder', QDir.homePath(), QFileDialog.ShowDirsOnly)
		if path != '':
			path += '/'
			self.saveDataAndVideo(path)
		else:
			self.statusBar.showMessage('Files were not saved... data is lost!')
		self.pauseResumeButton.setEnabled(False)
		self.stopSaveButton.setEnabled(False)
		self.startButton.setEnabled(False)
		self.loadButton.setEnabled(True)

	def getFilenames(self, path):
		# File counter
		i = 1
		# Get personal information
		subject = self.personalForm.getValues()
		# Create the file base name
		basename = '{}-{}-{}-{}'.format(subject['name'], subject['last'], subject['age'], subject['sex'])
		# Get current date
		date = datetime.today().strftime('%Y-%m-%d')
		# Add elapsed time and date
		basename += '_{}'.format(date)
		# Add file extension
		filename = basename + '_{}.csv'.format(i)
		# Complete the full path
		dataFullPath =  path + filename
		# Evaluate if the file name is already used
		while QFile(dataFullPath).exists():
			i += 1
			filename = basename + '_{}.csv'.format(i)
			dataFullPath =  path + filename
		# Create a new full path to store the video file
		filename = basename + '_{}.hdf5'.format(i)
		videoFullPath = path + filename
		# Return data and video full paths
		return (dataFullPath, videoFullPath)

	def saveDataAndVideo(self, path):
		# Get valid file names
		dataFullPath, videoFullPath = self.getFilenames(path)
		# Get the EEG data
		D = self.lsl.getData()
		# Get the EEG channels
		channels = self.connectionForm.getValues()['channels'].strip()
		channels = channels.split(',') if channels != '' else range(1, len(D[0]) + 1)
		# Create a new DataFrame
		file = pd.DataFrame(D, columns=channels)
		# Export DataFrame to CSV
		file.to_csv(dataFullPath, index=False)
		self.statusBar.showMessage('EEG file saved as {}'.format(dataFullPath))
		# Rename video file
		file = QFile('video.hdf5')
		file.rename(videoFullPath)
		self.statusBar.showMessage('Video file saved as {}'.format(videoFullPath))


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
		self.addButton.clicked.connect(self.addLabel)
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
		print('SetPos')
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
		numCh = param['channels']
		self.fs = param['fs']
		notch(self.X, self.fs, param['notch'], numCh)
		bandpass(self.X, self.fs, param['low'], param['high'], param['order'], numCh)
		self.X[self.X.columns[:numCh]] = MinMaxScaler().fit_transform(self.X[self.X.columns[:numCh]])
		self.label = np.array([0] * self.D.index.size)
		self.eegViewer.configure(channel, color, self.fs * 1.2, self.fs)
		self.eegViewer.plotData(self.X)
		self.eegViewer.setPosition(0)
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

	def addLabel(self):
		position = int(self.indexEdit.text().strip())
		self.eegViewer.addMark(position)
		self.label[position] = 1
		self.indexEdit.setText('')
		self.playButton.setFocus()

	def saveLabels(self):
		self.D['Label'] = self.label
		path = self.fileDialog.getFullPath()
		path = path.split('.')[0]
		path += '_Label.csv'
		self.D.to_csv(path, index=False)
		self.statusBar.showMessage('Labeled EEG file saved as {}'.format(path))


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
		self.X = None
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
		self.validationPanel = RadioButtonPanel(title='Action to execute')
		self.validationPanel.setOptions(['Validation', 'Prediction'])
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
		windowLayout.addWidget(QLabel('Step (%)'), 0, 3, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wstepEdit, 1, 3, alignment=Qt.AlignCenter)
		windowLayout.addWidget(QLabel('Padding (%)'), 0, 4, alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wpaddEdit, 1, 4, alignment=Qt.AlignCenter)
		windowGBox.setLayout(windowLayout)

		paramsLayout = QHBoxLayout()
		paramsLayout.addWidget(self.filteringForm, 6)
		paramsLayout.addWidget(windowGBox, 5)
		paramsLayout.addWidget(self.validationPanel, 2)

		mainLayout = QVBoxLayout()
		mainLayout.addLayout(viewLayout)
		mainLayout.addWidget(self.channelPanel)
		mainLayout.addLayout(paramsLayout)
		mainLayout.addWidget(self.applyButton, alignment=Qt.AlignCenter)
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
		columns = self.T.columns[:-1]
		self.channelPanel.setOptions(columns)

	def applyParameters(self):
		channels = self.channelPanel.getChecked()
		if len(channels) == 0:
			self.statusBar.showMessage('You must select at least one channel!')
			return
		# Data backup and preprocesing
		filters = self.filteringForm.getValues()
		# Training
		if self.T is not None:
			self.T_tmp = self.applyFilters(self.T, filters, channels)
		else:
			self.statusBar.showMessage('Training EEG file was not loaded correctly! Try again.')
			return
		# Validation
		if self.V is not None:
			self.V_tmp = self.applyFilters(self.V, filters, channels)
		# Window parameters
		self.param['channel'] = channels[0]
		self.param['wsize'] = int(self.wsizeEdit.text().strip())
		self.param['wover'] = float(self.woverEdit.text().strip())
		self.param['fixed'] = self.fixedCheck.isChecked()
		self.param['wstep'] = float(self.wstepEdit.text().strip())
		self.param['wpadd'] = float(self.wpaddEdit.text().strip())
		self.param['label'] = self.label_id
		# Reset labels and counters
		self.resetStates()
		# Now try to create the traininig and testing datasets
		self.createDatasets(channels)

	def applyFilters(self, D, filters, channels):
		C = D.copy()
		notch(C, filters['fs'], filters['notch'], filters['channels'])
		bandpass(C, filters['fs'], filters['low'], filters['high'], filters['order'], filters['channels'])
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
		valType = self.validationPanel.getChecked()
		# Verify if fixed windows will be used
		if self.param['fixed']:
			maxWsize = max_window_size(self.T_tmp, self.label_id)
			if self.param['wsize'] > maxWsize:
				self.statusBar.showMessage('There are some fixed windows overlaped. Check the window size.')
		self.T_tmp, self.y_train = extract_windows(self.T_tmp, self.param['wsize'], self.label_id, self.param['wover'], self.param['fixed'], self.param['wstep'], self.param['wpadd'])
		# Check validation type to create training and testing dataset 
		if valType == 'Validation' and self.V is not None and self.label_id in self.V.columns:
			# Copy for custom validation
			self.X = self.V_tmp.copy()
			self.V_tmp, self.y_test = extract_windows(self.V_tmp, self.param['wsize'], self.label_id, self.param['wover'], False, 0.0, 0.0)
			self.action = valType
		elif valType == 'Prediction' and self.V is not None:
			self.V_tmp, self.y_test = extract_windows(self.V_tmp, self.param['wsize'], self.label_id, self.param['wover'], False, 0.0, 0.0)
			self.action = valType
		else:
			# self.T_tmp, self.V_tmp, self.y_train, self.y_test = train_test_split(self.T_tmp, self.y_train, test_size=0.3)
			self.statusBar.showMessage('There was an error with {} EEG file.'.format(valType))
			return
		# Map lists of windows to numpy arrays
		self.X_train = np.array([win[channels[0]].values for win in self.T_tmp])
		self.y_train = np.array(self.y_train)
		self.X_test = np.array([win[channels[0]].values for win in self.V_tmp])
		self.y_test = np.array(self.y_test)
		# Message of dataset creation
		self.statusBar.showMessage('Datasets are ready!')
		# EEG viewer setup
		self.plotViewer['left'].configure(channels, colors, self.param['wsize'])
		self.plotViewer['right'].configure(channels, colors, self.param['wsize'])
		# Set plot counter
		self.plotTotal['left'] = len(self.X_train)
		# Plot first training window
		self.updatePlot('left', 1)
		# Activate run button
		self.runButton.setEnabled(True)

	def runExperiment(self):
		neuralnet = MLP_NN()
		# Input and output dimension
		input_dim = self.param['wsize']
		output_dim = 1
		# Ratio for pyramid shape in hidden layers
		ratio_io = int((input_dim / output_dim) ** (1 / 3))
		# Hidden layer dimension
		hidden_1 = output_dim * (ratio_io ** 2)
		hidden_2 = output_dim * ratio_io
		# Layer configuration
		layer = [hidden_1, hidden_2, output_dim]
		# Activation functions for each layer
		activation = ['relu', 'relu', 'sigmoid']
		# Optimizer for learning
		optimizer = 'adam'
		# Loss function for learning
		loss = 'mean_squared_error'
		# Required metrics
		metrics = ['accuracy']
		self.statusBar.showMessage('Building neural network...')
		neuralnet.build(input_dim, layer, activation, optimizer, loss, metrics)
		self.statusBar.showMessage('Running training phase...')
		neuralnet.training(self.X_train, self.y_train, val_size=0.3, epochs=100)
		# Get results
		if self.action == 'Prediction':
			score = neuralnet.prediction(self.X_test)
			self.y_pred = score
			self.updateStats(score, 'prediction')
		else:
			score = neuralnet.my_validation(self.X, self.param['wsize'], self.param['wover'], self.param['channel'], self.label_id)
			self.V_tmp, self.y_test, self.y_pred = score['X_test'], score['y_test'], score['y_pred']
			self.updateStats(score, 'validation')
		self.statusBar.showMessage('Neural network has finished!')
		# Set plot counter
		self.plotTotal['right'] = len(self.X_test)
		# Plot first testing window
		self.updatePlot('right', 1)

	def updatePlot(self, side, val):
		if (self.plotIndex[side] + val) >= 0 and (self.plotIndex[side] + val) < self.plotTotal[side]:
			self.plotIndex[side] += val
			data = self.T_tmp[self.plotIndex[side]] if side == 'left' else self.V_tmp[self.plotIndex[side]]
			start = data.index[0]
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
			if len(self.y_test) > 0: pixReal = self.pixmap[color[self.y_test[self.plotIndex[side]]]]['real']
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
			summary += '-> Accuracy: {}\n'.format(round(score['acc'], 5))
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