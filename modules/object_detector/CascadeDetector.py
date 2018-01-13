"""
verision: alpha
update: 1-13-2018
author: Aramisbuildtoys
"""
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

test_dir = './pos/'
test_format = '.jpeg'

class CascadeDetector():
    """cascade detector"""
    def __init__(self, mode="camera"):
        if mode not in ["camera", "image"]:
            raise Exception("mode should be 'camera' or 'image'!")
        self.mode = mode
        if self.mode == "camera":
            self.capture = cv2.VideoCapture(0)
        self.detector = cv2.CascadeClassifier('cascade.xml')
        self.detector.load('./gen/cascade.xml')
        
    def detect_camera(self):
        """detect target from camera"""
        if self.mode not in ["camera"]:
            raise Exception("not in camera mode!")
        while(True):
            ret, frame = self.capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            targets = self.detector.detectMultiScale(gray, 1.1, 5)
            img = frame.copy()
            
            for (x, y, w, h) in targets:
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_gray = gray[x:x+w, y:y+h]
                roi_color = img[x:x+w, y:y+h]
            cv2.imshow('detector test', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cv2.destoryAllWindows()
        
    def release_camera(self):
        """release camera capture"""
        if self.mode not in ["camera"]:
            raise Exception("not in camera mode!")
        self.capture.release()
    
    def detect_image(self, img_dir):
        """detect target from image file"""
        print "testing from image files"
        if self.mode not in ["image"]:
            raise Exception("not in image mode!")
        img_itor = glob.iglob(img_dir+'*'+test_format)
        for img_name in img_itor:
            
            frame = cv2.imread(img_name)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            targets = self.detector.detectMultiScale(gray, 1.3, 5)
            print "*"
            img = frame.copy()
            
            for (x, y, w, h) in targets:
                img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                roi_gray = gray[x:x+w, y:y+h]
                roi_color = img[x:x+w, y:y+h]
            cv2.imshow('img',img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
            #plt.gray()
            #plt.imshow(frame)
            #plt.show()
            #tmp = raw_input()
        
    
if __name__ == '__main__':
    #detector = CascadeDetector()
    #detector.detect_camera()
    #detector.release_camera()
    detector = CascadeDetector("image")
    detector.detect_image(test_dir)