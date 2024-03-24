import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def load_images_from_folder(folder):
    Blist = []
    Glist = []
    Rlist= []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            B,G,R=cv2.split(img)
            Blist.append(B)
            Glist.append(G)
            Rlist.append(R)
    return Blist,Glist,Rlist 

B,G,R=load_images_from_folder('img/new')
#B=np.array(B)
#G=np.array(G)
#R=np.array(R)

Bf=[]
for b in B:
    Bf.append(b.flatten())

Bc=np.concatenate(Bf).ravel()

Gf=[]
for g in G:
    Gf.append(g.flatten())

Gc=np.concatenate(Gf).ravel()

Rf=[]
for r in R:
    Rf.append(r.flatten())

Rc=np.concatenate(Rf).ravel()



Bmean,Bstd=cv2.meanStdDev(Bc)
Gmean,Gstd=cv2.meanStdDev(Gc)  
Rmean,Rstd=cv2.meanStdDev(Rc)
#b1,b2=cv2.meanStdDev(B)
#g1,g2=cv2.meanStdDev(G)
#r1,r2=cv2.meanStdDev(R)
print(f'Bmean: {Bmean} std: {Bstd}')
print(f'Gmean: {Gmean} std: {Gstd}')
print(f'Rmean: {Rmean} std: {Rstd}')
#print(f'Bmean: {b1} std: {b2}')
#print(f'Gmean: {g1} std: {g2}')
#print(f'Rmean: {r1} std: {r2}')
cv2.waitKey(0)  



