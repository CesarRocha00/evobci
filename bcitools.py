import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
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

	content = ['250','60','1','12','5']
	fields = ['fs','notch','low','high','order']
	labels = ['Fs (Hz)','Notch (Hz)','Low Cut (Hz)','Hight Cut (Hz)','Order']

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

	def setChannels(self, names):
		self.reset()
		# Create and add new controls
		row = 0
		colLimit = self.columns
		for i, label in enumerate(names):
			self.color[label] = '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))
			channel = QCheckBox(label)
			channel.setStyleSheet('background: {}'.format(self.color[label]))
			channel.stateChanged.connect(self.newChange)
			self.layout.addWidget(channel, row, i % colLimit, alignment=Qt.AlignCenter)
			self.control.append(channel)
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
			normal = QRadioButton(label)
			normal.toggled.connect(self.newChange)
			self.layout.addWidget(normal, alignment=Qt.AlignCenter)
			self.control.append(normal)
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


class EEGPlotter(QThread):
	"""docstring for EEGPlotter"""
	
	StoppedState = 0
	PausedState = 1
	RunningState = 2

	def __init__(self, mode='single', rows=4):
		super(EEGPlotter, self).__init__()
		self.mode = mode
		self.rows = rows
		self.view = GraphicsLayoutWidget()
		self.state = self.StoppedState
		self.plotItem = list()
		self.plotTrace = dict()
		# Holders
		self.wsize = None
		self.delay = None
		self.color = None
		self.window = None
		self.channel = None

	def widget(self):
		return self.view

	def show(self):
		self.view.show()

	def hide(self):
		self.view.hide()

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
		self.clear()
		self.plotItem.clear()
		self.plotTrace.clear()
		# Create new canvas
		if self.mode == 'single':
			self.singleLayout()
		else:
			self.multipleLayout()

	def plotData(self, D):
		for ch in self.channel:
			self.plotTrace[ch].setData(D[ch])

	def singleLayout(self):
		canvas = self.view.addPlot(0, 0)
		canvas.setAntialiasing(True)
		canvas.setClipToView(True)
		canvas.setDownsampling(mode='peak')
		canvas.disableAutoRange()
		# canvas.setRange(yRange=[0,1])
		for ch in self.channel:
			pen = mkPen(color=self.color[ch], width=2)
			self.plotTrace[ch] = canvas.plot(pen=pen)
		self.plotItem.append(canvas)

	def multipleLayout(self):
		col = 0
		rowLimit = self.rows
		for i, ch in enumerate(channel):
			pen = mkPen(color=self.color[ch], width=2)
			canvas = self.view.addPlot(i % rowLimit, col)
			canvas.setAntialiasing(True)
			canvas.setClipToView(True)
			canvas.setDownsampling(mode='peak')
			canvas.disableAutoRange()
			# canvas.setRange(yRange=[0,1])
			self.plotItem.append(canvas)
			self.plotTrace[ch] = canvas.plot(pen=pen)
			if (i + 1) % rowLimit == 0:
				col += 1

	def findIndex(self, index):
		index -= 1
		hsize = self.wsize / 2
		self.window[0] = index - hsize
		self.window[1] = index + hsize
		self.update()

	def update(self):
		self.window += 1
		for plot in self.plotItem:
			plot.setRange(xRange=self.window)

	def play(self):
		self.state = self.RunningState

	def pause(self):
		self.state = self.PausedState

	def toggle(self):
		self.state = self.PausedState if self.state == self.RunningState else self.RunningState

	def stop(self):
		self.state = self.StoppedState
		self.quit()

	def run(self):
		while True:
			if self.state == self.RunningState:
				self.update()
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
		self.player.mediaStatusChanged.connect(self.mediaStatusTrigger)
		self.player.error.connect(self.mediaError)
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
		self.statusBar.showMessage('¡System ready!')
		self.setStatusBar(self.statusBar)

	def loadVideo(self):
		path = self.videoDialog.getFullPath()
		self.player.loadMedia(path)
		self.camera.setSource(0)
		self.startButton.setEnabled(True)

	def mediaStatusTrigger(self, status):
		print(status)
		print('Player state:', self.player.state())
		print('Media status:', self.player.mediaStatus())
		if status == self.player.LoadingMedia:
			self.statusBar.showMessage('Loading video...')
		elif status == self.player.LoadedMedia:
			self.player.show()
			self.statusBar.showMessage('Video was successfully loaded!')
		elif status == self.player.EndOfMedia:
			self.stopAndSave()

	def mediaError(self):
		self.statusBar.showMessage('Error: {}'.format(self.player.errorString()))

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
		if channels != '':
			channels = channels.split(',')
		else:
			channels = range(1, len(D[0]) + 1)
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
		# Main widgets
		self.fileDialog = FilePathDialog('EEG file:', 'Open file', 'EEG files (*.csv)')
		self.videoDialog = FilePathDialog('Video file:', 'Open video', 'Video files (*.mp4)')
		self.channelPanel = CheckBoxPanel(title='Channels')
		self.filteringForm = FilteringForm(title='Filtering')
		# Video player
		self.player = VideoPlayer()
		self.player.setSize(640, 480)
		self.player.error.connect(self.mediaError)
		# Threads
		self.eegView = EEGPlotter(mode='single')
		# Tools
		self.scaler = MinMaxScaler()
		# EditLines
		self.wsizeEdit = QLineEdit('250')
		self.wsizeEdit.setAlignment(Qt.AlignCenter)
		self.timeEdit = QLineEdit('--:--:--')
		self.timeEdit.setAlignment(Qt.AlignCenter)
		self.indexEdit = QLineEdit('0')
		self.indexEdit.setAlignment(Qt.AlignCenter)
		# Buttons
		self.loadButton = QPushButton('Load')
		self.saveButton = QPushButton('Save')
		self.addButton = QPushButton('Add')
		self.applyButton = QPushButton('Apply')
		self.playButton = QPushButton('Play')
		self.stopButton = QPushButton('Stop')
		self.findTimeButton = QPushButton('Time')
		self.findIndexButton = QPushButton('Index')
		# Slider
		self.positionSlider = QSlider(Qt.Horizontal)
		self.positionSlider.setRange(0, 1)
		# GroupBoxes
		windowGBox = QGroupBox('Window')
		windowGBox.setAlignment(Qt.AlignCenter)
		searchGBox = QGroupBox('Search')
		searchGBox.setAlignment(Qt.AlignCenter)
		labelingGBox = QGroupBox('Labeling')
		labelingGBox.setAlignment(Qt.AlignCenter)
		# Layouts
		sourceLayout = QGridLayout()
		sourceLayout.setVerticalSpacing(0)
		sourceLayout.addWidget(self.fileDialog, 0, 0)
		sourceLayout.addWidget(self.videoDialog, 1, 0)
		sourceLayout.addWidget(self.loadButton, 0, 1, 2, 1)

		windowLayout = QVBoxLayout()
		windowLayout.addWidget(QLabel('Num. samples'), alignment=Qt.AlignCenter)
		windowLayout.addWidget(self.wsizeEdit, alignment=Qt.AlignCenter)
		windowGBox.setLayout(windowLayout)

		paramsLayout = QHBoxLayout()
		paramsLayout.addWidget(self.filteringForm, 8)
		paramsLayout.addWidget(windowGBox, 1)
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
		self.statusBar.showMessage('¡System ready!')
		self.setStatusBar(self.statusBar)

	def loadVideo(self):
		path = self.videoDialog.getFullPath()
		self.player.loadMedia(path)
		# self.startButton.setEnabled(True)

	def mediaError(self):
		self.statusBar.showMessage('Error: {}'.format(self.player.errorString()))


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
sys.exit(app.exec())
