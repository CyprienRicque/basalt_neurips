import json
import logging
import os

import av
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


# Create a PyQt5 window with a video player widget, a timeline widget, and a label list widget
class VideoLabeler(QtWidgets.QMainWindow):
    def __init__(self, video_path):
        super().__init__()

        # Load the video and retrieve its metadata
        self.video_path = video_path
        self.container = av.open(video_path)
        self.video_stream = next(s for s in self.container.streams if s.type == 'video')
        self.duration = self.container.duration / self.video_stream.time_base
        self.frame_rate = self.video_stream.average_rate
        self.num_frames = int(self.duration * self.frame_rate)

        # Create the video player widget
        self.video_player = VideoPlayerWidget(self.container, self.video_stream)

        # Create the timeline widget
        self.timeline = TimelineWidget(self)
        self.timeline.setMinimumHeight(100)
        self.timeline.setMaximumHeight(100)

        # Create the label list widget
        self.label_list = QtWidgets.QListWidget()
        self.label_list.setMinimumWidth(200)
        self.label_list.setMaximumWidth(200)

        # Wrap the label list widget in a dock widget
        self.label_list_dock = QtWidgets.QDockWidget()
        self.label_list_dock.setWidget(self.label_list)

        # Add the widgets to the window
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_player)
        layout.addWidget(self.timeline)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_list_dock)

        # Create a dictionary to store the labels for each frame
        self.labels = {}

        # Set the window title and size
        self.setWindowTitle(f'Video Labeler - {os.path.basename(video_path)}')
        self.resize(800, 600)

        # Connect the signals and slots
        self.video_player.position_changed.connect(self.timeline.update_position)
        self.timeline.position_changed.connect(self.video_player.seek)
        self.label_list.itemDoubleClicked.connect(self.seek_to_label)

    # Seek to the frame corresponding to a label when it is clicked in the label list widget
    def seek_to_label(self, item):
        frame_index = int(item.text().split(':')[0])
        self.video_player.seek(frame_index)

    # Handle key press events to label the frames and navigate the videos
    def keyPressEvent(self, event):
        # Get the current frame index and label
        frame_index = self.video_player.current_frame
        label = event.text()
        # If the key corresponds to a label, add the label to the labels dictionary and update the label list widget
        if label in ['t', 's', 'p']:
            if frame_index in self.labels:
                self.labels[frame_index].add(label)
            else:
                self.labels[frame_index] = {label}

            self.update_label_list()

        # If the key is 'n', go to the next video
        elif label == 'n':
            self.next_video()

        # If the key is 'p', go to the previous video
        elif label == 'p':
            self.prev_video()

        # If the key is 'd', delete the label for the current frame
        elif label == 'd':
            self.labels.pop(frame_index, None)
            self.update_label_list()

        # If the key is 's', save the labels to a jsonl file
        elif label == 's':
            self.save_labels()

    # Update the label list widget with the current labels
    def update_label_list(self):
        self.label_list.clear()
        for frame_index, labels in self.labels.items():
            label_str = ', '.join(labels)
            item = QtWidgets.QListWidgetItem(f'{frame_index}: {label_str}')
            self.label_list.addItem(item)

    # Go to the next video in the video list
    def next_video(self):
        pass  # TODO: Implement this

    # Go to the previous video in the video list
    def prev_video(self):
        pass  # TODO: Implement this

    # Save the labels to a jsonl file
    def save_labels(self):
        with open('labels.jsonl', 'w') as f:
            for frame_index, labels in self.labels.items():
                label_dict = {'frame_index': frame_index, 'labels': list(labels)}
                json.dump(label_dict, f)
                f.write('\n')


# Create a custom video player widget that uses PyAV to play the video
class VideoPlayerWidget(QtWidgets.QWidget):
    position_changed = QtCore.pyqtSignal(int)

    def __init__(self, container, video_stream):
        super().__init__()

        self.container = container
        self.video_stream = video_stream
        self.current_frame = 0
        self.total_frames = int(self.container.duration / self.video_stream.time_base)
        self.frame_rate = self.video_stream.average_rate
        self.playback_speed = 1.0

        # Create a label to display the video frames
        self.label = QtWidgets.QLabel(self)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Create a timer to update the video frame at regular intervals
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.frame_rate))

        # Set the layout for the widget
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    # Update the video frame displayed in the label
    def update_frame(self):
        # Seek to the current frame
        self.container.seek(int(self.current_frame * self.video_stream.time_base), any_frame=True)

        # Retrieve the next video frame
        for packet in self.container.demux():
            for frame in packet.decode():
                # print(object_methods)
                # print(type(frame), frame)
                if isinstance(frame, av.video.frame.VideoFrame):
                    # Convert the frame to a numpy array and update the label with the image
                    image = frame.to_ndarray()
                    # print("ici", image.shape, image)
                    assert image.shape == (540, 640)
                    image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QtGui.QImage.Format_RGB888)
                    if image is None:
                        logging.error("QImage could not be created")
                    else:
                        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

                    # image = frame.to_image()
                    # image_data = np.frombuffer(image.planes[0], np.uint8)
                    # image = image_data.reshape(image.height, image.width, image.format.num_planes)
                    # image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * image.shape[2],
                    #                      QtGui.QImage.Format_RGB888)
                    # self.label.setPixmap(QtGui.QPixmap.fromImage(image))


                    # Increment the current frame index and break
                    self.current_frame += self.playback_speed
                    break

        # Emit the position changed signal
        self.position_changed.emit(int(self.current_frame))

    # Seek to a specific point in the video
    def seek(self, frame_index):
        self.current_frame = frame_index
        self.update_frame()


# Create a custom timeline widget that displays the video timeline and allows the user to seek to a specific point
class TimelineWidget(QtWidgets.QWidget):
    position_changed = QtCore.pyqtSignal(int)

    def __init__(self, video_labeler):
        super().__init__()

        self.video_labeler = video_labeler
        self.duration = self.video_labeler.duration
        self.frame_rate = self.video_labeler.frame_rate
        self.num_frames = self.video_labeler.num_frames

        # Create a label to display the frame preview
        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setFixedSize(100, 100)
        self.preview_label.setVisible(False)

        # Create a plot widget to display the timeline
        self.plot = pg.PlotWidget(self)
        self.plot.setXRange(0, self.duration)
        self.plot.setMouseTracking(True)
        self.plot.showAxis('bottom', False)
        self.plot.hideButtons()
        self.plot.setMaximumHeight(100)

        # Set the layout for the widget
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.preview_label)
        self.setLayout(layout)

    # Update the position of the timeline cursor and the frame preview when the video position changes
    def update_position(self, frame_index):
        # Update the position of the timeline cursor
        self.plot.clear()
        self.plot.plot([frame_index / self.frame_rate, frame_index / self.frame_rate], [0, 1],
                       pen=pg.mkPen('w', width=2))

        # Update the frame preview
        self.video_labeler.container.seek(int(frame_index * self.video_labeler.video_stream.time_base),
                                          any_frame=True)
        for packet in self.video_labeler.container.demux():
            for frame in packet.decode():
                if isinstance(frame, av.video.frame.VideoFrame):
                    # Convert the frame to a numpy array and update the preview label with the image
                    # image = frame.to_image()
                    # image_data = np.frombuffer(image.planes[0], np.uint8)
                    # image = image_data.reshape(image.height, image.width, image.format.num_planes)
                    # image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                    #                      image.shape[1] * image.shape[2],
                    #                      QtGui.QImage.Format_RGB888)
                    # self.preview_label.setPixmap(QtGui.QPixmap.fromImage(image))

                    # Convert the frame to a numpy array and update the label with the image
                    image = frame.to_ndarray()
                    # print("ici2", image.shape, image)
                    assert image.shape == (540, 640)
                    image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QtGui.QImage.Format_RGB888)
                    if image is None:
                        logging.error("QImage could not be created")
                    else:
                        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(image))

                    break

    # Seek to a specific point in the video when the user clicks on the timeline
    def mousePressEvent(self, event):
        # Calculate the frame index from the x position of the mouse click
        x = self.plot.plotItem.vb.mapSceneToView(event.pos()).x()
        frame_index = int(x * self.frame_rate)

        # Seek to the frame and emit the position changed signal
        self.position_changed.emit(frame_index)

    # Show the frame preview when the mouse is moved over the timeline
    def mouseMoveEvent(self, event):
        # Calculate the frame index from the x position of the mouse
        x = self.plot.plotItem.vb.mapSceneToView(event.pos()).x()
        frame_index = int(x * self.frame_rate)

        # Update the position of the timeline cursor and the frame preview
        self.update_position(frame_index)
        self.preview_label.setVisible(True)


# Create the main window and display the video labeling GUI
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = VideoLabeler("../basalt_neurips_data/BuildWaterFall/woozy-ruby-ostrich-ff3ddf992ab4-20220721-125851.mp4")
    window.show()
    app.exec_()