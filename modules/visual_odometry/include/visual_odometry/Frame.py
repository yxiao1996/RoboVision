from Camera import *
from util import *
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

class Frame():
    def __init__(self, color, depth=None):
        self.id = id
        self.time_stamp = time.time()
        self.color = color
        self.depth = depth
        self.camera = Camera()
        self.keypoints, self.descriptors = detectORB(color)

    def getK(self):
        return self.camera.camera_matrix['data']
    
    def getFocalPp(self):
        K = np.array(self.camera.camera_matrix['data']).reshape([3,3])
        focal = np.array((K[0,0], K[1,1]))
        pp = (K[0,2], K[1,2])
        return focal, pp
    
    def getRt(self):
        R = self.camera.R
        t = self.camera.t
        return R, t
    
    def setRt(self, R, t):
        self.camera.R = R
        self.camera.t = t

    def plotFeature(self):
        img2 = np.array([])
        img3 = cv2.drawKeypoints(self.color, self.keypoints, color=(0, 255, 0), outImage=img2)
        # display image
        cv2.imshow('imagewithfeature', img3)
        #plt.gray()
        #plt.imshow(img3)
        #plt.show()
if __name__ == '__main__':
    img = cv2.imread('./0.jpeg', )
    f = Frame(img)
    img1 = cv2.imread('./2.jpeg', )
    f1 = Frame(img)
    matches = matchBF(f.descriptors, f1.descriptors)
    plot_matches(f.color, f.keypoints, f1.color, f1.keypoints, matches)