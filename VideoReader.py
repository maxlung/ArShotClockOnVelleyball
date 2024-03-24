from threading import Thread
import time
import cv2
import queue
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.buffer=queue.Queue()
    def getSource(self,src):
        self.stream = cv2.VideoCapture(src)
    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        
        while not self.stopped:
            if (not self.grabbed) and self.buffer.empty:
                self.stop()
            elif(self.buffer.qsize()<60):
                (self.grabbed, self.frame) = self.stream.read()
                self.stream.grab()
                self.buffer.put(self.frame) 
            time.sleep(0.01)
    def pop(self):
        print(self.buffer.qsize())
        if not self.buffer.empty():
            frame=self.buffer.get()
            print('not empty')
        else:
            frame=self.frame
            print('empty')
        return frame

    def stop(self):
        self.stream.release()
        self.stopped = True