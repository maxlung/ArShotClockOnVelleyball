import cv2
import numpy as np
import time
from VideoReader import VideoGet
from VideoShower import VideoShow
from PIL import Image, ImageDraw, ImageFont

drawing = False # true if mouse is pressed
font = ImageFont.truetype('Lato-BlackItalic.ttf', 50)
src_x, src_y = -1,-1
kp_list = []
kp2_list=[[0,0],[400,0],[400,800],[0,800]]
output_width, output_height=1280,720
source="./video/HDR80_D_Live_20230212_160854_000.MOV"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (output_width,output_height))

def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def draw_shotClock():
    num_image = Image.new('RGBA', (400,800),(0, 0, 0, 0))
    draw = ImageDraw.Draw(num_image)
    #text
    text = f"{countdown}"
    #Positioning Text
    textwidth, textheight = draw.textsize(text, font)
    width, height = num_image.size
    x=width/2-textwidth/2
    y=height-textheight-40
    
    #Applying text on image via draw object
    draw.text((x, y), text,font=font,fill=(255, 255, 255, 0)) 

    return num_image

def find_mask(img):
    Bmean,Bstd=149.194,7.6
    Gmean,Gstd= 182.3,3.722
    Rmean,Rstd=255,7
    image=img.copy()
    B,G,R=cv2.split(image)
    mask=(~((abs(B-Bmean)>(3*Bstd))&(abs(G-Gmean)>(3*Gstd))&(abs(R-Rmean)>(3*Rstd)))).astype(np.uint8)
    return mask


def combineWithAlpha(foreground):
    # normalize alpha channels from 0-255 to 0-1

    alpha_foreground = foreground[:,:,3].astype(float) / 255.0
    alpha_foreground= np.stack((alpha_foreground,alpha_foreground,alpha_foreground), axis=2)
    foreground=cv2.cvtColor(foreground,cv2.COLOR_BGRA2BGR).astype(float)
    # set adjusted colors
    """ 
    for color in range(0, 3):
        foreground[:,:,color] = alpha_foreground * foreground[:,:,color]
    """
    
    foreground=cv2.multiply(foreground,alpha_foreground)
    foreground=foreground.astype(np.uint8)
    
    # set adjusted alpha and denormalize back to 0-255
    #background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255


    return foreground


vidcap= cv2.VideoCapture(source)
success, image =vidcap.read()
cv2.namedWindow('main')
cv2.setMouseCallback('main', select_points_src)


while(success):
    frame=cv2.resize(image,(output_width,output_height))
    cv2.imshow('main',frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        print('save points')
        kp_list.append([src_x,src_y])
        print("src points:")
        print(kp_list)    
    elif k == 27:
        cv2.destroyAllWindows()
        vidcap.release()
        break





#multitreading image get
video_reader = VideoGet(source).start()
video_shower = VideoShow(image).start()
countdown=8
count_time=0
num_image = draw_shotClock()#draw num 8
success=True
#geometrical transformation
H, m = cv2.findHomography(np.array(kp_list).reshape(-1,1,2), np.array(kp2_list).reshape(-1,1,2), cv2.RANSAC,5.0)
while success:
    success,image=video_reader.grabbed,video_reader.pop()
    startTime = time.time()
    frame=cv2.resize(image,(output_width,output_height))
    
    #key points
    for kp in kp_list:
        cv2.circle(frame,kp,3,(0,0,255),-1)
    transformed_frame=cv2.warpPerspective(frame,H,(400,800))
    #people block problem
    mask=find_mask(transformed_frame)
    #draw shot clock
    transformed_frame=Image.fromarray(cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGBA))
    #Creating draw object
   
    
    #Creating text and font object
    if(count_time>1 and countdown>0):
        num_image=draw_shotClock()
        countdown=countdown-1
        count_time=0
    
    #people block problem
    num_image=np.array(num_image)
    num_image.setflags(write=1)
    num_image[:,:,3]=mask*255
    num_image=Image.fromarray(num_image)
   

    #convert to cv2 image format
    transformed_frame.paste(num_image, (0, 0),mask=num_image.convert('RGBA'))
    transformed_frame=cv2.cvtColor(np.array(transformed_frame), cv2.COLOR_RGBA2BGRA)
    
    num_image_cv=cv2.cvtColor(np.array(num_image), cv2.COLOR_RGBA2BGRA)
    num_image_cv=cv2.warpPerspective(num_image_cv, np.linalg.inv(H) ,(output_width,output_height))
    
    
    #imshow("num",num_image_cv)
    
    #combine two image
   
    
    num_image_cv=combineWithAlpha(num_image_cv) 
    frame=cv2.addWeighted(frame,1,num_image_cv,1,-1)   
    
    #im_thresh_gray = cv2.bitwise_or(frame, frame, mask=mask)   
    #cv2.imshow('Mask',im_thresh_gray) //this is for test mask
    
    #for save video
    write_image=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
    
    #out.write(write_image)
    
    #cv2.imshow("main", frame)
    video_shower.frame=frame

    endTime = time.time() - startTime
    count_time=endTime+count_time
    #cv2.imshow("transformed",transformed_frame)
    if video_shower.stopped==True:
        video_reader.stop()
        break
    if(video_reader.stopped):
        break
    
    print(endTime)

out.release()
