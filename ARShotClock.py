import cv2
import numpy as np
import time
from VideoReader import VideoGet
from VideoShower import VideoShow
from PIL import Image, ImageDraw, ImageFont

class ArShotclock:
    def __init__(self,source):
        self.drawing = False # true if mouse is pressed
        self.font = ImageFont.truetype('Lato-BlackItalic.ttf', 60)
        self.src_x, self.src_y = -1,-1
        self.kp_list = []
        self.kp2_list=[[0,0],[400,0],[400,800],[0,800]]
        self.output_width, self.output_height=1280,720
        self.source=source
        self.draw_x=185
        self.draw_y=710
        self.text_width, self.text_height=35,60
        #self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #self.out = cv2.VideoWriter('output.avi', fourcc, 60.0, (output_width,output_height))

    def select_points_src(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            self.src_x, self.src_y = x,y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def getKps(self):
        self.kp_list = []
        vidcap= cv2.VideoCapture(self.source)
        success, image =vidcap.read()
        cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.select_points_src)
        while(success):
            frame=cv2.resize(image,(self.output_width,self.output_height))
            cv2.imshow('main',frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                print('save points')
                self.kp_list.append([self.src_x,self.src_y])
                print("src points:")
                print(self.kp_list)    
            elif k == 27:
                cv2.destroyAllWindows()
                vidcap.release()
                self.H, m = cv2.findHomography(np.array(self.kp_list).reshape(-1,1,2), np.array(self.kp2_list).reshape(-1,1,2), cv2.RANSAC,5.0)
                self.invH=np.linalg.inv(self.H)
                break
            elif k == ord('d'):
                cv2.destroyAllWindows()
                vidcap.release()
                self.kp_list = [[835,361], [1099,456], [205,624], [76,474]]
                self.H, m = cv2.findHomography(np.array(self.kp_list).reshape(-1,1,2), np.array(self.kp2_list).reshape(-1,1,2), cv2.RANSAC,5.0)
                self.invH=np.linalg.inv(self.H)
                break
    
    def builsMaps(self,srcsize,H):
        # Since the camera won't be moving, pregenerate the remap LUT

        # Generate the warp matrix
        srcTM = H.copy(); # If WARP_INVERSE, set srcTM to transformationMatrix

        map_x=np.empty(shape=(srcsize[1],srcsize[0]),dtype=np.float32)
        map_y=np.empty(shape=(srcsize[1],srcsize[0]),dtype=np.float32)

        M11 = srcTM[0,0]
        M12 = srcTM[0,1]
        M13 = srcTM[0,2]
        M21 = srcTM[1,0]
        M22 = srcTM[1,1]
        M23 = srcTM[1,2]
        M31 = srcTM[2,0]
        M32 = srcTM[2,1]
        M33 = srcTM[2,2]

        for y in range(srcsize[1]):
            fy = float(y)
            for  x in range(srcsize[0]):
                fx = float(x)
                w = (M31 * fx) + (M32 * fy) + M33
                if(w!=0):
                    w=1/w
                else:
                    w=0
                new_x = (float)((M11 * fx) + (M12 * fy) + M13) * w
                new_y = (float)((M21 * fx) + (M22 * fy) + M23) * w
                map_x[y,x] = new_x
                map_y[y,x] = new_y
            
        

        # This creates a fixed-point representation of the mapping resulting in ~4% CPU savings
        #map_x, map_y=cv2.convertMaps(map_x, map_y,dstmap1type=cv2.CV_16SC2, nninterpolation=False)
        return map_x, map_y

    
    def draw_shotClock(self,num):
        num_image = Image.new('RGBA', (400,800),(0, 0, 0, 0))
        draw = ImageDraw.Draw(num_image)
        #text
        text = f"{num}"
        #Positioning Text
        self.text_width, self.text_height = draw.textsize(text, self.font)
        '''
        width, height = num_image.size
        x=width/2-textwidth/2
        y=height-textheight-40
        print('textheight:',textheight)
        print('textwidth:',textwidth)
        print('x:',x)
        print('y:',y)
        '''
        
        #Applying text on image via draw object
        draw.text((self.draw_x, self.draw_y), text,font=self.font,fill=(255, 255, 255, 0)) 

        return num_image

    def find_mask(self,img):
        Bmean,Bstd=149.194,7.6
        Gmean,Gstd= 182.3,3.722
        Rmean,Rstd=255,3
        image=img.copy()
        B,G,R=cv2.split(image)
        mask=(~((abs(B-Bmean)>(1*Bstd))&(abs(G-Gmean)>(1*Gstd))&(abs(R-Rmean)>(0.5*Rstd)))).astype(np.uint8)
        return mask


    def combineWithAlpha(self,foreground):
        # normalize alpha channels from 0-255 to 0-1
        
        alpha_foreground = foreground[:,:,3].astype(float) / 255.0
        alpha_foreground= np.stack((alpha_foreground,alpha_foreground,alpha_foreground), axis=2)
        foreground=cv2.cvtColor(foreground,cv2.COLOR_BGRA2BGR).astype(float)
        
        # set adjusted colors
        """ 
        for color in range(0, 3):
            foreground[:,:,color] = alpha_foreground * foreground[:,:,color]
        """
        start=time.time()
        foreground=cv2.multiply(foreground,alpha_foreground)
        foreground=foreground.astype(np.uint8)
        end=time.time()-start
        # set adjusted alpha and denormalize back to 0-255
        #background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        print('alpha:',end)
        return foreground
    
    def quickcombineWithAlpha(self,foreground):
        img = Image.fromarray(foreground)
        background = Image.new('RGBA', img.size, (0,0,0))
        alpha_composite = Image.alpha_composite(background, img)
        return alpha_composite

    
    def Process_ShotClock(self,frame,num):
        startTime = time.time()
        frame=cv2.resize(frame,(self.output_width,self.output_height))
        transformed_frame=cv2.warpPerspective(frame,self.H,(400,800))
        
        #people block problem
        
        MaskROI=transformed_frame[self.draw_y:self.text_height+self.draw_y,self.draw_x:self.text_width+self.draw_x]
        #print(transformed_frame.shape)
        #print(MaskROI.shape)
        mask=self.find_mask(MaskROI)
        #draw shot clock
        #transformed_frame=Image.fromarray(cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGBA))
        
        num_image=self.draw_shotClock(num)
        
        
        #people block problem
        num_image=np.array(num_image)
        num_image.setflags(write=1)
        num_image[self.draw_y:self.text_height+self.draw_y,self.draw_x:self.text_width+self.draw_x,3]=mask*255
        num_image=self.quickcombineWithAlpha(num_image)
        #convert to cv2 image format
        #transformed_frame.paste(num_image, (0, 0),mask=num_image.convert('RGBA'))
        #transformed_frame=cv2.cvtColor(np.array(transformed_frame), cv2.COLOR_RGBA2BGRA)
        
        num_image_cv=cv2.cvtColor(np.array(num_image), cv2.COLOR_RGBA2BGR)
        num_image_cv=cv2.warpPerspective(num_image_cv, self.invH ,(self.output_width,self.output_height))

        '''
        warped_Numimg=np.zeros((self.output_height,self.output_width,4), dtype=np.uint8)
        cv2.remap(num_image_cv, dst=warped_Numimg, map1=self.mapX, map2=self.mapY, interpolation=cv2.INTER_LINEAR)
        print(warped_Numimg.shape,type(warped_Numimg))
        print(frame.shape,type(frame))
        print(self.mapX.shape,self.mapY.shape)
        cv2.imshow("num",warped_Numimg)
        '''
        
        #combine two image
        
        #num_image_cv=self.combineWithAlpha(num_image_cv) 
        
        

        frame=cv2.addWeighted(frame,1,num_image_cv,1,-1)   
        
        #im_thresh_gray = cv2.bitwise_or(frame, frame, mask=mask)   
        #cv2.imshow('Mask',im_thresh_gray) //this is for test mask
        
        #for save video
        write_image=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        
        #out.write(write_image)
        
        #cv2.imshow("main", frame)

        
        #cv2.imshow("transformed",transformed_frame)
        endTime = time.time() - startTime
        print(endTime)
        
        return write_image