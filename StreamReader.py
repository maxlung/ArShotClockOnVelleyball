import ffmpeg
import numpy as np
from threading import Thread
import time
import cv2
class StreamGet:

    def __init__(self, src=r'rtsp://nckusport:Ncku1234@10.30.3.31:554/stream0'):
        probe = ffmpeg.probe(src)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])

        self.out = (
            ffmpeg
                .input(src, rtsp_transport='tcp')
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=30)
                .run_async(pipe_stdout=True)
        )
        self.cnt_empty = 0
        
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.in_bytes=None
        self.frame=None
        self.isStreamming = False

    def start(self):    
        self.thread.start()
    
    def update(self):
        while True:
            a=time.time()
            self.in_bytes = self.out.stdout.read(self.height * self.width * 3)
            

            if self.cnt_empty > 10:
                break
            self.cnt_empty = 0
            
            if not self.in_bytes:
                self.cnt_empty += 1
            else:
                self.frame = np.frombuffer(self.in_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
            
            self.isStreamming=True
            b=time.time()
            print(b-a)
            # to process frame
    def get(self):
        return self.frame
            
    def stop(self):
        self.stopped = True
'''
Stream=StreamGet()
Stream.start()


while True:
    a=time.time()
    frame=Stream.get()
    # to process frame
    try:
        if(Stream.isStreamming):cv2.imshow('test',frame)
    except AttributeError:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    b=time.time()
    #print(b-a)
'''