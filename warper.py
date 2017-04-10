import cv2
import numpy as np
import matplotlib.pyplot as plt 


class Warper:
    def __init__(self):
        self.src = []
        self.ratio = []
        
        
    def check_src(self):
# Check and adjust trapezoid if lane change to the right is occuring 
        if self.src[2,0]<= int(self.ratio[0,0] * 640):
            temp = self.src[2,0]
            self.src[2,0] = self.ratio[0,0] * 1280 - self.src[3,0]
            self.src[3,0] = temp
            temp = self.src[1,0]
            self.src[1,0] = self.ratio[0,0] * 1280 - self.src[0,0]
            self.src[0,0] = temp
            #print('Lane change to right')
            
# Check and adjust trapezoid if lane change to the left is occuring
        elif self.src[3,0] >= int(self.ratio[0,0] * 640):
            temp = self.src[3,0]
            self.src[3,0] = self.ratio[0,0] * 1280 - self.src[2,0]
            self.src[2,0] = temp
            temp = self.src[0,0]
            self.src[0,0] = self.ratio[0,0] * 1280 - self.src[1,0]
            self.src[1,0] = temp
            #print('Lane change to left')
            
    def set_transforms(self,img):
        self.ratio = np.float32([[img.shape[1]/1280, 0], [0, img.shape[0]/720]])   
                
        dst = np.float32([
            [260 , 0],
            [1040 , 0],
            [1040 , 720],
            [260 , 720],
            ])
        dst = np.matmul(dst,self.ratio)  
#Check if src points are available, if not => use GUI to select src points
        try:
            src = np.load('data/src.npy')
        #Check if any changes have been made to stored points
            if (src != self.src).any():
                self.src = src
                self.check_src()
                self.M = cv2.getPerspectiveTransform(self.src, dst)
                self.Minv = cv2.getPerspectiveTransform(dst, self.src)      
                        
        except:
#Show image and expected point locations and ask for user input to get new src points
            plt.imshow(img)
            src = np.float32([
                [580 , 460],
                [700 , 460],
                [1040, 680],
                [260 , 680],
                ])    
            src = np.matmul(src,self.ratio)
            plt.scatter(src[:,0],src[:,1],c='b',s=100,alpha=0.3)
            plt.title('Select FOUR best trapezoid points \n Starting from TOP-LEFT corner in CLOCKWISE manner \n (RIGHT CLICK to cancel last selected point)') 
            self.src =  plt.ginput(4)
            plt.close()
        #operate on selected points to make them usable
            self.src = np.asarray(self.src,dtype=np.uint32)
            self.src[0:2,1] = (self.src[0,1] + self.src[1,1])/2
            self.src[2:4,1] = (self.src[2,1] + self.src[3,1])/2
            self.src = np.float32(self.src)
        #Calculate transforms and save the src points
            self.M = cv2.getPerspectiveTransform(self.src, dst)
            self.Minv = cv2.getPerspectiveTransform(dst, self.src)     
            np.save('data/src.npy',self.src)
            
    def warp(self, img):
        self.set_transforms(img)
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        self.set_transforms(img)
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )
