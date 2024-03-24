from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog 
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from ARShotClock import ArShotclock
import cv2
import qimage2ndarray
from VideoReader import VideoGet
from StreamReader import StreamGet
import time


class Main:

    def __init__(self):
        self.ui = QUiLoader().load('mainwindow.ui')
        self.frame_timer = QtCore.QTimer()
        QtCore.QTimer.setTimerType(self.frame_timer,QtCore.Qt.PreciseTimer)
        self.countdown_timer = QtCore.QTimer()
        QtCore.QTimer.setTimerType(self.countdown_timer, QtCore.Qt.PreciseTimer)
        self.fps=30
        self.state=0#0=ncku,1=logo,2=countdown
        self.min=5
        self.sec=0
        self.scoreA=0
        self.scoreB=0
        self.nameA=''
        self.nameB=''
        self.filepath='./video/ptz32.mkv'#"./video/HDR80_D_Live_20230212_160854_000.MOV"
        self.ui.Button_start.clicked.connect(self.Start)
        self.countdown_timer.timeout.connect(self.countDown)
        self.ui.Button_quit.clicked.connect(self.Quit)
        self.ui.Button_kp.clicked.connect(self.select_kp)
        self.ui.Button_countdown.clicked.connect(self.StartCountDown)
        self.ui.Button_block.clicked.connect(self.ShowBlock)
        self.ui.Button_file.clicked.connect(self.Openfile)
        self.ui.Button_ncku.clicked.connect(self.Startncku)
        self.ui.Button_logo.clicked.connect(self.Startlogo)
        self.ui.Button_Aup.clicked.connect(self.Aup)
        self.ui.Button_Adown.clicked.connect(self.Adown)
        self.ui.Button_Bup.clicked.connect(self.Bup)
        self.ui.Button_Bdown.clicked.connect(self.Bdown)
        self.started=False
        self.playing=False
        self.isShowingBlock=False
        self.ARShotClock=ArShotclock(source=self.filepath)
        self.video_reader = VideoGet(self.filepath)
        #self.stream_reader = StreamGet()
        #self.stream_reader.start()

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
        
    def Openfile(self):
        self.filepath,temp=QFileDialog.getOpenFileName(QMainWindow(),"選擇影片")
        print(self.filepath)
        self.ARShotClock.source=self.filepath
        self.video_reader.getSource(self.filepath)
        
        

    def select_kp(self):
        self.ARShotClock.getKps(self.video_reader.pop())

    def Quit(self):
        self.video_reader.stop()
        QApplication.quit()
    def ShowBlock(self):
        if(self.isShowingBlock==False):
            self.isShowingBlock=True
        else:
            self.isShowingBlock=False

    def displayFrame(self):
        startTime = time.time()
        #ret, frame = self.video_reader.grabbed,self.video_reader.pop()
        #if(not ret):
        #   print('not successful')
        #frame=self.stream_reader.get()
        frame=self.video_reader.pop()

        frame=self.ARShotClock.Process_ShotClock(frame,self.min,self.sec,self.state,self.nameA,self.scoreA,self.nameB,self.scoreB,self.isShowingBlock)
        frame=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)
        cv2.imshow("demo",frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = qimage2ndarray.array2qimage(frame)
        #self.ui.label_img.setPixmap(QtGui.QPixmap.fromImage(image))
        endTime = time.time() - startTime
        print('total:',endTime)
        interval=(int(1000//self.fps)-int(endTime*1000)) if (int(1000//self.fps)-int(endTime*1000))>0 else 1
        self.frame_timer.setInterval(interval)
        print('interval:',int(1000//self.fps)-int(endTime*1000))
    

    def Startncku(self):
        self.state=0
    def Startlogo(self):
        self.state=1
        self.nameA=self.ui.input_nameA.text()
        self.nameB=self.ui.input_nameB.text()
        self.scoreA=int(self.ui.input_A.text())
        self.scoreB=int(self.ui.input_B.text())
    def Aup(self):
        self.scoreA+=1
        self.ui.input_A.setText(f'{self.scoreA}')
    def Adown(self) :
        self.scoreA-=1
        self.ui.input_A.setText(f'{self.scoreA}')
    def Bup(self):
        self.scoreB+=1
        self.ui.input_B.setText(f'{self.scoreB}')
    def Bdown(self):
        self.scoreB-=1
        self.ui.input_B.setText(f'{self.scoreB}')

        



    def StartCountDown(self):
        self.min=int(self.ui.input_min.text())
        self.sec=int(self.ui.input_sec.text())
        self.countdown_timer.start(1000)
        self.state=2
        

    def countDown(self):
        if self.min>=0:
            self.sec-=1
            if self.sec==-1 :
                self.sec=59
                self.min-=1
                if(self.min==-1 or self.state==0):
                    self.state=0
                    self.countdown_timer.stop()
        else:
            self.state=0
            self.countdown_timer.stop()
app = QApplication([])
stats = Main()
stats.ui.show()
app.exec_()
