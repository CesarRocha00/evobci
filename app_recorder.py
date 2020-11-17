import pandas as pd
from datetime import datetime
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QDir, QFile
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStatusBar
from custom_widgets import FilePathDialog, LSLForm, PersonalInformationForm, VideoPlayer, LSLClient, OCVWebcam

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


import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	view = EEGRecorder()
	view.show()
	sys.exit(app.exec_())