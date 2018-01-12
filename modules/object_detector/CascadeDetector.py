"""
verision: alpha
update: 12-6-2017
author: Aramisbuildtoys
"""
import numpy as np
import cv2

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
    
    def detect_image(self, img_name):
        """detect target from image file"""
        if self.mode not in ["image"]:
            raise Exception("not in image mode!")
        
    
if __name__ == '__main__':
    detector = CascadeDetector()
    detector.detect_camera()
    detector.release_camera()
