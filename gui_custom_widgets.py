import cv2
import h5py
import numpy as np
from time import sleep
from datetime import datetime
from pylsl import StreamInlet, resolve_stream
from PyQt5.QtCore import Qt, QDir, QUrl, QThread
from pyqtgraph import mkPen, GraphicsLayoutWidget
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QPushButton, QWidget, QGroupBox, QLineEdit, QStyle, QFileDialog, QRadioButton, QFormLayout, QGridLayout, QCheckBox

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
	fields = ['fs','notch','low','high','order','n_channels']
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
			self.color[label] = f'#{np.random.randint(0x000000, 0xFFFFFF):06x}'
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

	def addMark(self, position, label=None):
		for canvas in self.plotItem:
			pen = mkPen(color='g', width=2.5, style=Qt.DashLine)
			hpen = mkPen(color='r', width=2.5, style=Qt.DashLine)
			mark = canvas.addLine(x=position, pen=pen, label=label, labelOpts={'position':0.9}, movable=True, hoverPen=hpen)
			return mark

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