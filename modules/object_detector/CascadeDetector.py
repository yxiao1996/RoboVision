"""
verision: alpha
update: 1-13-2018
author: Aramisbuildtoys
"""
import numpy as np
import cv2
import glob
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from IoU import CalcuIoU as IOU

test_dir = './pos/'
test_format = '.jpeg'
anno_dir = '/home/yxiao1996/data/balls/Annotations/pos/'

def getItor(anno_dir):
    itor = glob.iglob(anno_dir+'*.xml')
    
    return itor

def getRoot(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    return root

def findXYWH(root):
    bun_box = root.find('object').find('bndbox')
    max_x = float(bun_box.find('xmax').text)
    min_x = float(bun_box.find('xmin').text)
    max_y = float(bun_box.find('ymax').text)
    min_y = float(bun_box.find('ymin').text)
    x = int((max_x + min_x) / 2)
    y = int((max_y + min_y) / 2)
    width = int(max_x - min_x)
    height = int(max_y - min_y)

    return x, y, width, height

class CascadeDetector():
    """cascade detector"""
    def __init__(self, mode="camera"):
        if mode not in ["camera", "image"]:
            raise Exception("mode should be 'camera' or 'image'!")
        self.mode = mode
        if self.mode == "camera":
            self.capture = cv2.VideoCapture(0)
        self.detector = cv2.CascadeClassifier('cascade.xml')
        #self.detector.load('./gen/cascade.xml')
        self.detector.load('/home/yxiao1996/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        
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
    
    def detect_image(self, root):
        """detect target from image file"""
        print "testing from image files"
        if self.mode not in ["image"]:
            raise Exception("not in image mode!")
        img_itor = glob.iglob(img_dir+'*'+test_format)
        for img_name in img_itor:
            
            frame = cv2.imread(img_name)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            targets = self.detector.detectMultiScale(gray, 1.3, 20)
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
    
    def test_image(self, xml_dir):
        path = root.find('path').text
        X, W, W, H = findXYWH(root)

        rgb = cv2.imread(path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        targets = self.detector.detectMultiScale(gray, 1.3, 20)
        
        for (x, y, w, h) in targets:
            cv2.imshow('img',img)
            cv2.waitKey(0)
            print IOU(x, y, w, h, X, Y, W, H)
            raw_input()
    
if __name__ == '__main__':
    #detector = CascadeDetector()
    #detector.detect_camera()
    #detector.release_camera()
    detector = CascadeDetector("image")
    # detector.detect_image(test_dir)
    filename_itor = getItor(anno_dir)
    for filename in filename_itor:
        root = getRoot(filename)
        detector.test_image(root)
    cv2.destroyAllWindows()
