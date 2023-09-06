from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from ARShotClock import ArShotclock
import cv2
import qimage2ndarray
from VideoReader import VideoGet
from VideoShower import VideoShow
import time


class Main:

    def __init__(self):
        self.ui = QUiLoader().load('mainwindow.ui')
        self.frame_timer = QtCore.QTimer()
        QtCore.QTimer.setTimerType(self.frame_timer,QtCore.Qt.PreciseTimer)
        self.countdown_timer = QtCore.QTimer()
        QtCore.QTimer.setTimerType(self.countdown_timer, QtCore.Qt.PreciseTimer)
        self.fps=30
        self.isCount=False
        self.num=8
        self.ui.Button_start.clicked.connect(self.Start)
        self.countdown_timer.timeout.connect(self.countDown)
        self.ui.Button_quit.clicked.connect(self.Quit)
        self.ui.Button_kp.clicked.connect(self.select_kp)
        self.ui.Button_countdown.clicked.connect(self.StartCountDown)
        self.ARShotClock=ArShotclock(source="./video/HDR80_D_Live_20230212_160854_000.MOV")
        self.video_reader = VideoGet("./video/HDR80_D_Live_20230212_160854_000.MOV")
        self.started=False
        self.playing=False
        

    def Start(self):
        if(self.playing==False):
            if(self.started==False):
                self.video_reader.start()
                
                self.frame_timer.timeout.connect(self.displayFrame)
            self.frame_timer.start(int(1000 // self.fps))
            self.ui.Button_start.setText("Pause")
            self.started=True
            self.playing=True
        else:
            self.frame_timer.stop()
            self.ui.Button_start.setText("Start")
            self.playing=False
        

    def select_kp(self):
        self.ARShotClock.getKps()

    def Quit(self):
        self.video_reader.stop()
        QApplication.quit()
    
    def displayFrame(self):
        startTime = time.time()
        ret, frame = self.video_reader.grabbed,self.video_reader.pop()
        if(not ret):
           print('not successful')
        if self.isCount:
            frame=self.ARShotClock.Process_ShotClock(frame,self.num)
        else:
            frame=cv2.resize(frame,(1280,720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(frame)
        self.ui.label_img.setPixmap(QtGui.QPixmap.fromImage(image))
        endTime = time.time() - startTime
        print('total:',endTime)
        interval=(int(1000//self.fps)-int(endTime*1000)) if (int(1000//self.fps)-int(endTime*1000))>0 else 1
        self.frame_timer.setInterval(interval)
        print('interval:',int(1000//self.fps)-int(endTime*1000))

    def StartCountDown(self):
        self.countdown_timer.start(1000)
        self.isCount=True
        self.num=8

    def countDown(self):
        if self.num>0:
            self.num=self.num-1
        else:
            self.isCount=False
            self.countdown_timer.stop()
app = QApplication([])
stats = Main()
stats.ui.show()
app.exec_()
