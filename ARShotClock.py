import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from numba import jit,njit
import numba as nb
class ArShotclock:
    def __init__(self,source):
        self.drawing = False # true if mouse is pressed
        self.font = ImageFont.truetype('Lato-BlackItalic.ttf', 60)
        self.font2 = ImageFont.truetype('GenRyuMin-B.ttc', 35)
        self.src_x, self.src_y = -1,-1
        self.kp_list = []
        self.output_width, self.output_height=854,480#1280,720#1920,1080
        self.kp2_list=[[0,0],[400,0],[400,800],[0,800]]
        
        self.source=source
        self.draw_x=150
        self.draw_y=710
        self.text_width, self.text_height=35,60

        self.logoncku=Image.open('./logo/ncku3.png').convert('RGBA')
        self.logoncku=self.logoncku.resize([120,120])
        self.logoA=Image.open('./logo/caesar.png').convert('RGBA')
        self.logoA=self.logoA.resize([120,120])
        self.logoB=Image.open('./logo/tpWhale.png').convert('RGBA')
        self.logoB=self.logoB.resize([120,120])
        #self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #self.out = cv2.VideoWriter('output.avi', fourcc, 60.0, (output_width,output_height))

    def select_points_src(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            self.src_x, self.src_y = x,y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def getKps(self,frame):
        self.kp_list = []
        #vidcap= cv2.VideoCapture(self.source)
        #success, image = vidcap.read()
        image=frame
        cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.select_points_src)
        while(1):
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
                #vidcap.release()
                self.H, m = cv2.findHomography(np.array(self.kp_list).reshape(-1,1,2), np.array(self.kp2_list).reshape(-1,1,2), cv2.RANSAC,5.0)
                self.invH=np.linalg.inv(self.H)
                self.min_x = self.max_x = self.min_y = self.max_y = self.kp_list[0]
                for point in self.kp_list[1:]:
                    if point[0] < self.min_x[0]:
                        self.min_x = point
                    if point[0] > self.max_x[0]:
                        self.max_x = point
                    if point[1] < self.min_y[1]:
                        self.min_y = point
                    if point[1] > self.max_y[1]:
                        self.max_y = point
                print(self.min_x, self.max_x, self.min_y, self.max_y)
                break
            elif k == ord('d'):
                cv2.destroyAllWindows()
                #vidcap.release()
                self.kp_list = [[838, 365], [1096, 456], [199, 618], [73, 474]]
                self.H, m = cv2.findHomography(np.array(self.kp_list).reshape(-1,1,2), np.array(self.kp2_list).reshape(-1,1,2), cv2.RANSAC,5.0)
                self.invH=np.linalg.inv(self.H)
                self.min_x = self.max_x = self.min_y = self.max_y = self.kp_list[0]
                for point in self.kp_list[1:]:
                    if point[0] < self.min_x[0]:
                        self.min_x = point
                    if point[0] > self.max_x[0]:
                        self.max_x = point
                    if point[1] < self.min_y[1]:
                        self.min_y = point
                    if point[1] > self.max_y[1]:
                        self.max_y = point
                break
    
    
    
    def draw_shotClock(self,min,sec,state,nameA,scoreA,nameB,scoreB,isblock):
        num_image = Image.new('RGBA', (400,800),(0, 0, 0, 0))
        draw = ImageDraw.Draw(num_image)
        #Positioning Text
        #self.text_width, self.text_height = draw.textsize(text, self.font)
        
        if(state==0):
            draw.text((self.draw_x-50, self.draw_y), 'N C K U',font=self.font,fill=(255, 255, 255, 255)) 
            num_image.paste(self.logoncku,((int)(400/3),400+(int)(400/3)),self.logoncku)
        elif(state==1):
            draw.text((self.draw_x-70, self.draw_y), f'{scoreA}',font=self.font,fill=(255,255, 255, 255)) 
            draw.text((self.draw_x+90, self.draw_y), f'{scoreB}',font=self.font,fill=(107,255 ,117, 255)) 
            draw.text((self.draw_x-140, self.draw_y-150), f'{nameA}',font=self.font2,fill=(255,255, 255, 255)) 
            draw.text((self.draw_x+90, self.draw_y-150), f'{nameB}',font=self.font2,fill=(107,255 ,117, 255)) 
            #num_image.paste(self.logoA,((int)(400/3),400+(int)(400/3)),self.logoA)

            #draw.text((self.draw_x+30,10 ), f'{scoreB}',font=self.font2,fill=(255, 255, 255, 255)) 
            #num_image.paste(self.looB,((int)(400/3),100),self.logoB)
        elif(state==2):
            #count down text
            if(sec>9):
                text = f"{min} : {sec}"
            else:
                text = f"{min} : 0{sec}"
            #Applying text on image via draw object
            draw.text((self.draw_x, self.draw_y), text,font=self.font,fill=(255, 255, 255, 255)) 
            
        
        

            
        if(isblock):
            color1,color2=(0,150,100, 255),(125,100,0, 255)
            draw.line([(0, 800/6), (400,  800/6)], fill =color1, width = 3)
            draw.line([(0, 800*2/6), (400,  800*2/6)], fill =color1, width = 3)
            draw.line([(400/3, 0), (400/3,  400)], fill =color1, width = 3)
            draw.line([(400*2/3, 0), (400*2/3,  400)], fill =color1, width = 3)

            draw.line([(0, 800*4/6), (400,  800*4/6)], fill =color2, width = 3)
            draw.line([(0, 800*5/6), (400,  800*5/6)], fill =color2, width = 3)
            draw.line([(400/3, 400), (400/3,  800)], fill =color2, width = 3)
            draw.line([(400*2/3, 400), (400*2/3,  800)], fill =color2, width = 3)
        return num_image
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def find_mask(img):
        Bmean,Bstd=89.45,6.67
        Gmean,Gstd= 148.11,7.66
        Rmean,Rstd=253.34,3.26

        #B,G,R=cv2.split(img)
        B,G,R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask=(170*(~((np.abs(B-Bmean)>(2*Bstd))&(np.abs(G-Gmean).astype(np.uint8)>(2*Gstd))&(np.abs(R-Rmean).astype(np.uint8)>(2*Rstd)))).astype(np.uint8))

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
    def overlayFrame(self,frame,img):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 在image2上覆盖image1的非零值部分
        frame[np.where(img[:,:]!=[0,0,0])] = img[np.where(img[:,:]!=[0,0,0])]
       
        
        return frame
    
    @staticmethod
    @jit(nopython=True)  
    def overlayWithAlpha(frame,img):
        alpha_channel = img[:, :, 3] # convert from 0-255 to 0.0-1.0
        ROI_overlay_colors = img[:, :, :3]
        
        #cv2.imshow("test",img[:, :, 3])
        #cv2.imshow("test2",frame)
        #alpha_mask =alpha_channel[:,:,np.newaxis] 
        alpha_mask =np.dstack((alpha_channel, alpha_channel, alpha_channel))
        frame =   (frame* (1 - alpha_mask/255)).astype('uint8') +(ROI_overlay_colors * (alpha_mask/255)).astype('uint8')
        
        #composite1=cv2.multiply(frame,(1 - alpha_mask/255),dtype=cv2.CV_8UC3)
        #composite2=cv2.multiply(overlay_colors, alpha_mask/255,dtype=cv2.CV_8UC3)
        
        return frame #cv2.add(composite1,composite2)
    

    
    def Process_ShotClock(self,frame,min,sec,state,nameA,scoreA,nameB,scoreB,isShowingBlock):
        """
        Do something
        Args:
            frame:
            ...

        output:

        comment
        """
        
        frame=cv2.resize(frame,(self.output_width,self.output_height))

        
        transformed_frame=cv2.warpPerspective(frame,self.H,(400,800))
        
        #people block problem
        
        #MaskROI=transformed_frame[self.draw_y:self.text_height+self.draw_y,self.draw_x:self.text_width+self.draw_x]
        #print(transformed_frame.shape)
        #print(MaskROI.shape)
        
        #draw shot clock
        #transformed_frame=Image.fromarray(cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGBA))
        
        num_image=self.draw_shotClock(min,sec,state,nameA,scoreA,nameB,scoreB,isblock=isShowingBlock)
        mask=self.find_mask(transformed_frame)
        
        #people block problem
        num_image=np.array(num_image)
        num_image.setflags(write=1)
        #num_image[self.draw_y:self.text_height+self.draw_y,self.draw_x:self.text_width+self.draw_x,3]=mask*255
        num_image[:,:,3]=mask
        #num_image=self.quickcombineWithAlpha(num_image)
        
        
        
        
        num_image_cv=cv2.cvtColor(np.array(num_image), cv2.COLOR_RGBA2BGRA)
        num_image_cv=cv2.warpPerspective(num_image_cv, self.invH ,(self.output_width,self.output_height))

        alpha = num_image_cv[:, :, 3]
        alpha[np.all(num_image_cv[:, :, 0:3] == (0, 0, 0), 2)] = 0
        #combine two image
        
        #frame=self.overlayFrame(frame=frame,img=num_image_cv)
        #frame=cv2.addWeighted(frame,1,num_image_cv,1,-1)
        startTime = time.time()   
        frame[self.min_y[1]:self.max_y[1],self.min_x[0]:self.max_x[0]]=self.overlayWithAlpha(frame[self.min_y[1]:self.max_y[1],self.min_x[0]:self.max_x[0]],num_image_cv[self.min_y[1]:self.max_y[1],self.min_x[0]:self.max_x[0]])#overlay
        endTime = time.time() - startTime
        
        #im_thresh_gray = cv2.bitwise_or(transformed_frame, transformed_frame, mask=mask)   
        #cv2.imshow('Mask',im_thresh_gray) #this is for test mask
        

        
        
        #cv2.imshow("main", frame)

        
        #cv2.imshow("transformed",transformed_frame)
        
        print(endTime)
        
        return frame
'''    
ar=ArShotclock("./video/HDR80_D_Live_20230212_160854_000.MOV")
ar.getKps()
img=ar.draw_shotClock(0,False,True)

img.show()
'''