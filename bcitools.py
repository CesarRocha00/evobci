import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from filters import notch, bandpass
from pyqtgraph import mkPen, GraphicsLayoutWidget
from sklearn.preprocessing import MinMaxScaler
from pylsl import StreamInlet, resolve_stream
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import (Qt, QDir, QUrl, QFile, QTime, QTimer, QThread,
						  pyqtSignal as Signal, pyqtSlot as Slot)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QStatusBar, QVBoxLayout,
							 QHBoxLayout, QPushButton, QWidget, QGroupBox, QLineEdit,
							 QSizePolicy, QSlider, QStyle, QFileDialog, QRadioButton,
							 QFormLayout, QGridLayout, QDesktopWidget, QCheckBox)

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
		if title != None:
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
		if title != None:
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
		if title != None:
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
		if title != None:
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
		if title != None:
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
		self.wsize = 0
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

	def configure(self, channel, color, wsize=None):
		# Link params
		self.wsize = wsize
		self.color = color
		self.channel = channel
		if wsize != None:
			hsize = wsize / 2
			self.window = np.array([-hsize, hsize])
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
		canvas.setDownsampling(mode='peak')
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
			canvas.setDownsampling(mode='peak')
			canvas.showGrid(x=True, y=True, alpha=0.25)
			self.plotItem.append(canvas)
			self.plotTrace[ch] = canvas.plot(pen=pen)
			if (i + 1) % rowLimit == 0:
				col += 1

	def plotData(self, D):
		for ch in self.channel:
			self.plotTrace[ch].setData(D[ch])
		self.position = 0
		self.maxPosition = D.index.size

	def addMark(self, position):
		for canvas in self.plotItem:
			canvas.addLine(x=position)

	def setPosition(self, position):
		hsize = self.wsize / 2
		self.window[0] = position - hsize
		self.window[1] = position + hsize
		self.position = position
		self.update()

	def update(self):
		for plot in self.plotItem:
			plot.setRange(xRange=self.window, yRange=[0.0,1.0])
		self.position += 1 if self.position < self.maxPosition else 0

	def play(self):
		self.state = self.RunningState

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


class OCVWebcam(QThread):
	"""docstring for OCVWebcam"""

	StoppedState = 0
	PausedState = 1
	RecordingState = 2

	def __init__(self):
		super(OCVWebcam, self).__init__()
		self.F = list()
		self.source = None
		self.state = self.StoppedState

	def setSource(self, src):
		self.source = cv2.VideoCapture(src)
		self.source.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		self.F.clear()

	def getFrames(self):
		return self.F

	def record(self):
		self.state = self.RecordingState

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
				_, frame = self.source.read()
				self.F.append(frame)
			elif self.state == self.PausedState:
				pass
			else:
				break
		self.source.release()


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
		self.setWindowTitle('EEG Recorder')
		self.resize(480,640)
		self.setContentsMargins(10, 10, 10, 10)
		# Elapsed time holder
		self.elapsed = 0
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
		self.camera.setSource(0)
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
		self.lsl.record()
		self.lsl.start()
		self.camera.record()
		self.camera.start()
		self.elapsed = datetime.now()
		if self.player.isVideoAvailable():
			self.player.play()
		self.statusBar.showMessage('Recording...')
		self.pauseResumeButton.setEnabled(True)
		self.stopSaveButton.setEnabled(True)
		self.startButton.setEnabled(False)
		self.loadButton.setEnabled(False)

	def pauseAndResume(self):
		if self.lsl.getState() == LSLClient.PausedState:
			self.pauseResumeButton.setText('Pause')
			self.statusBar.showMessage('Recording...')
		else:
			self.pauseResumeButton.setText('Resume')
			self.statusBar.showMessage('Paused!')
		self.lsl.toggle()
		self.camera.toggle()
		if self.player.isVideoAvailable():
			self.player.toggle()

	def stopAndSave(self):
		self.lsl.stop()
		self.camera.stop()
		self.elapsed = datetime.now() - self.elapsed
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

	def getFileNames(self, path):
		# File counter
		i = 1
		# Get personal information
		subject = self.personalForm.getValues()
		# Create the file base name
		baseName = '{}-{}-{}-{}'.format(subject['name'], subject['last'], subject['age'], subject['sex'])
		# Get current date
		date = datetime.today().strftime('%Y-%m-%d')
		# Add elapsed time and date
		baseName += '_{}'.format(date)
		# Add file extension
		fileName = baseName + '_{}.csv'.format(i)
		# Complete the full path
		dataFullPath =  path + fileName
		# Evaluate if the file name is already used
		while QFile(dataFullPath).exists():
			i += 1
			fileName = baseName + '_{}.csv'.format(i)
			dataFullPath =  path + fileName
		# Create a new full path to store the video file
		fileName = baseName + '_{}.avi'.format(i)
		videoFullPath = path + fileName
		# Return data and video full paths
		return (dataFullPath, videoFullPath)

	def saveDataAndVideo(self, path):
		# Get valid file names
		dataFullPath, videoFullPath = self.getFileNames(path)
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
		# Get and count all recorded frames
		F = self.camera.getFrames()
		total_frames = len(F)
		# Compute the elapsed time
		total_seconds = self.elapsed.total_seconds()
		# Compute FPS
		fps = total_frames / total_seconds
		print('FPS:', fps)
		print('Duration: {} seconds'.format(total_seconds))
		# Set the video codec
		codec = cv2.VideoWriter_fourcc(*"XVID")
		# Prepare the output file writer
		file = cv2.VideoWriter(videoFullPath, codec, fps, (320, 240))
		# Write frames on file
		for frame in F:
			file.write(frame)
		file.release()
		self.statusBar.showMessage('Video file saved as {}'.format(videoFullPath))


class EEGLabeling(QMainWindow):
	"""docstring for EEGLabeling"""
	def __init__(self):
		super(EEGLabeling, self).__init__()
		self.setWindowTitle('EEG Labeling')
		self.setContentsMargins(10, 10, 10, 10)
		# Variables
		self.D = None
		self.X = None
		self.fs = 0
		self.wsize = 0
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
		self.eegViewer = EEGViewer(mode='multiple')
		# Tools
		self.scaler = MinMaxScaler()
		# EditLines
		self.wsizeEdit = QLineEdit('250')
		self.wsizeEdit.setAlignment(Qt.AlignCenter)
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
		windowGBox = QGroupBox('Window')
		windowGBox.setAlignment(Qt.AlignCenter)
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

		windowLayout = QVBoxLayout()
		windowLayout.addWidget(QLabel('Num. samples'), alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wsizeEdit, alignment=Qt.AlignCenter)
		windowGBox.setLayout(windowLayout)

		paramsLayout = QHBoxLayout()
		paramsLayout.addWidget(self.filteringForm, 12)
		paramsLayout.addWidget(windowGBox, 2)
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
			self.statusBar.showMessage('CSV file not found!')

	def mediaError(self):
		self.statusBar.showMessage('Error: {}'.format(self.player.errorString()))

	def setPosition(self, position):
		if self.player.isVideoAvailable():
			self.player.setPosition(position)
		else:
			self.eegViewer.setPosition(position)

	def durationChanged(self, duration):
		self.maxDuration = duration
		self.positionSlider.setRange(0, duration)
		self.timeEdit.setText('00:00:00')

	def positionChanged(self, position):
		self.positionSlider.setValue(position)
		h = (position / 3600000) % 24
		m = (position / 60000) % 60
		s = (position / 1000) % 60
		i = (position / 1000) * self.wsize
		self.eegViewer.setPosition(i)
		elapsed = QTime(h,m,s)
		self.timeEdit.setText(elapsed.toString())

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
		self.wsize = int(self.wsizeEdit.text().strip())
		numCh = param['channels']
		self.fs = param['fs']
		notch(self.X, self.fs, param['notch'], numCh)
		bandpass(self.X, self.fs, param['low'], param['high'], param['order'], numCh)
		self.X[self.X.columns[:numCh]] = self.scaler.fit_transform(self.X[self.X.columns[:numCh]])
		self.label = np.array([0] * self.D.index.size)
		self.eegViewer.configure(channel, color, self.wsize)
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
		self.playButton.setFocus()

	def stop(self):
		if self.player.isVideoAvailable():
			self.player.stop()
		self.playButton.setFocus()
		self.applyButton.setEnabled(True)

	def timeToSeconds(self):
		text = self.timeEdit.text()
		h, m, s = text.split(':')
		h, m, s = int(h), int(m), int(s)
		seconds = h * 3600 + m * 60 + s
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


class BCIVisualizer(QMainWindow):
	"""docstring for BCIVisualizer"""
	def __init__(self):
		super(BCIVisualizer, self).__init__()
		self.setWindowTitle('BCI Visualizer')
		self.resize(1280, 720)
		self.setContentsMargins(10, 10, 10, 10)

app = QApplication(sys.argv)
view = EEGLabeling()
view.show()
sys.exit(app.exec_())