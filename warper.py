import cv2
import numpy as np
import matplotlib.pyplot as plt 


class Warper:
    def __init__(self):
        self.src = None
        
            
    def set_transforms(self,img):
        ratio_x = img.shape[1] / 1280
        ratio_y = img.shape[0] / 720   
                
        dst = np.float32([
            [260 * ratio_x, 0 * ratio_y],
            [1040 * ratio_x, 0 * ratio_y],
            [1040 * ratio_x, 720 * ratio_y],
            [260 * ratio_x, 720 * ratio_y],
            ]) 
        try:
            src = np.load('data/src.npy')
            if src != self.src:
                self.src = src
                self.M = cv2.getPerspectiveTransform(self.src, dst)
                self.Minv = cv2.getPerspectiveTransform(dst, self.src)      
                        
        except:
            plt.imshow(img)
            src = np.float32([
                [580 * ratio_x, 460 * ratio_y],
                [700 * ratio_x, 460 * ratio_y],
                [1040 * ratio_x, 680 * ratio_y],
                [260 * ratio_x, 680 * ratio_y],
                ])    
            plt.hold
            plt.scatter(src[:,0],src[:,1],s=100,alpha=0.3)
            self.src =  plt.ginput(4)
            plt.close()
            self.src = np.asarray(self.src,dtype=np.uint32)
            self.src = np.float32(self.src)
            self.M = cv2.getPerspectiveTransform(self.src, dst)
            self.Minv = cv2.getPerspectiveTransform(dst, self.src)     
            
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
