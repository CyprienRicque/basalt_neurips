import os
import sys
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon

import logging
logging.basicConfig(level=logging.DEBUG)


class VideoPlayer(QMainWindow):

    def __init__(self, directory, parent=None):
        super(VideoPlayer, self).__init__(parent)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Create a horizontal box layout
        hBoxLayout = QHBoxLayout()
        hBoxLayout.setContentsMargins(0, 0, 0, 0)

        # Add widgets to the layout
        hBoxLayout.addWidget(self.playButton)
        hBoxLayout.addWidget(self.positionSlider)

        # Create a vertical box layout
        vBoxLayout = QVBoxLayout()
        vBoxLayout.addWidget(videoWidget)
        vBoxLayout.addLayout(hBoxLayout)
        vBoxLayout.addWidget(self.errorLabel)

        # Create a widget to hold the layout
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(vBoxLayout)

        # Set the central widget of the main window
        self.setCentralWidget(self.centralWidget)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        # Find the first MP4 file in the directory
        for file in os.listdir(directory):
            if file.endswith(".mp4"):
                path = os.path.join(directory, file)
                break

        # Load the video file
        media = QMediaContent(QUrl.fromLocalFile(path))
        self.mediaPlayer.setMedia(media)

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MP4 Files (*.mp4)", options=options)
        if fileName:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


# Create the main window and display the video labeling GUI
# if __name__ == '__main__':
#     app = QtWidgets.QApplication([])
#     window = VideoLabeler("../basalt_neurips_data/BuildWaterFall/woozy-ruby-ostrich-ff3ddf992ab4-20220721-125851.mp4")
#     window.show()
#     app.exec_()

if __name__ == '__main__':
    directory = "../basalt_neurips_data/BuildWaterFall/"
    app = QApplication(sys.argv)
    player = VideoPlayer(directory)
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())