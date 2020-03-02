import sys
from PyQt5.QtWidgets import QApplication
from bcitools import EEGRecorder, EEGLabeling, BCIVisualizer

app = QApplication(sys.argv)
if sys.argv[1] == 'rec':
	view = EEGRecorder()
	view.show()
elif sys.argv[1] == 'lab':
	view = EEGLabeling()
	view.show()
elif sys.argv[1] == 'bci':
	view = BCIVisualizer()
	view.show()
sys.exit(app.exec_())