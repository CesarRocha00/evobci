import cv2
import h5py
import time
import platform
import numpy as np
from datetime import datetime
from bcitools import OCVWebcam

# Best FPS: [key, data]
# Best compression: [key, data, maxshape=(frame.shape[0], frame.shape[1], frame.shape[2]), compression='gzip', chunks=True]

def video_2_hdf5(filename, frame_limit):
	# Create HDF5 file
	file = h5py.File(filename, 'w')
	# Open webcam
	cap = cv2.VideoCapture(0)
	# Set resolution
	width, height = 640, 480
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	# Frame counter
	total_frames = 0
	# Initialize timer
	time.sleep(0.5)
	elapsed = datetime.now()
	# Get frames
	while True:
		if cv2.waitKey(1) & 0xFF == ord('q') or total_frames == frame_limit:
			break
		ret, frame = cap.read()
		file.create_dataset('{}'.format(total_frames), data=frame)
		total_frames += 1
	# Stop timer
	elapsed = datetime.now() - elapsed
	total_seconds = elapsed.total_seconds()
	# Compute FPS
	fps = total_frames / total_seconds
	# Store attributes
	file.attrs['fps'] = fps
	file.attrs['width'] = width
	file.attrs['height'] = height
	file.attrs['length'] = total_frames
	file.attrs['duration'] = total_seconds
	# Release webcam
	cap.release()
	# Close HDF5 file
	file.close()
	# Print information
	print('Filename: {}\nFPS: {}\nFrames: {}\nSeconds: {}'.format(filename, fps, total_frames, total_seconds))

def hdf5_2_video(filename):
	# Get system OS
	os = platform.system()
	# Set codec and file extension
	codec = 'mp4v' if os == 'Darwin' else 'XVID'
	extension = '.mp4' if os == 'Darwin' else '.avi'
	outputfile = filename.split('/')[-1].split('.')[0] + extension
	fourcc = cv2.VideoWriter_fourcc(*codec)
	# Open HDF5 file
	file = h5py.File(filename, 'r')
	# Get attributes
	fps = file.attrs['fps']
	width = file.attrs['width']
	height = file.attrs['height']
	length = file.attrs['length']
	duration = file.attrs['duration']
	# Open video writer
	output = cv2.VideoWriter(outputfile, fourcc, fps, (width, height))
	# Write frames
	for i in range(length):
		key = '{}'.format(i)
		frame = file.get(key)
		frame = np.array(frame)
		output.write(frame)
	# Close video writer
	output.release()
	# Close HDF5 file
	file.close()
	# Print information
	print('Filename: {}\nFPS: {}\nFrames: {}\nSeconds: {}'.format(filename, fps, length, duration))

# filename = '../../../../Desktop/video.hdf5'
# video_2_hdf5(filename, 500)
# hdf5_2_video(filename)

filename = 'vid.hdf5'
cam = OCVWebcam()
cam.setSource(0)
cam.record(filename)
time.sleep(10)
cam.stop()
hdf5_2_video(filename)