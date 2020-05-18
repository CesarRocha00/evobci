import sys
from PyQt5.QtWidgets import QApplication
from bcitools import EEGRecorder, EEGStudio, BCISimulator

app = QApplication(sys.argv)
if sys.argv[1] == 'recorder':
	view = EEGRecorder()
	view.show()
elif sys.argv[1] == 'studio':
	view = EEGStudio()
	view.show()
elif sys.argv[1] == 'simulator':
	view = BCISimulator()
	view.show()
sys.exit(app.exec_())