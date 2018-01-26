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
anno_dir = '/home/yxiao1996/data/balls/Anno/'

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
            self.capture = cv2.VideoCapture(1)
        self.detector = cv2.CascadeClassifier('cascade.xml')
        self.detector.load('./gen/cascade.xml')
        #self.detector.load('/home/yxiao1996/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        
    def detect_camera(self):
        """detect target from camera"""
        if self.mode not in ["camera"]:
            raise Exception("not in camera mode!")
        while(True):
            ret, frame = self.capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            targets = self.detector.detectMultiScale(gray, 1.3, 6)
            img = frame.copy()
            
            for (x, y, w, h) in targets:
                img = cv2.rectangle(img, (x-w/2,y-h/2), (x+w/2,y+h/2), (255,0,0), 2)
                roi_gray = gray[x:x+w, y:y+h]
                roi_color = img[x:x+w, y:y+h]
                break
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
    
    def test_image(self, root, consec=False): 
        path = root.find('path').text
        X, Y, W, H = findXYWH(root)

        rgb = cv2.imread(path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        targets = self.detector.detectMultiScale(gray, 1.2, 3)
        
        for (x, y, w, h) in targets:
            img = cv2.rectangle(rgb, (x,y), (x+w,y+h), (255,0,0), 2)
            img = cv2.rectangle(rgb, (X-W/2,Y-H/2), (X+W/2,Y+H/2), (0,255,0), 2)
            #cv2.waitKey(0)
            x = x + w/2
            y = y + h/2
            print IOU(x, y, w, h, X, Y, W, H)
            if consec:
                return IOU(x, y, w, h, X, Y, W, H)

            while(1):
                cv2.imshow('img',img)
                key = cv2.waitKey(20)
                if key & 0xFF == ord(" "):
                    break
            break
    
    def test(self):
        preciesion_0 = 0
        preciesion_1 = 0
        preciesion_2 = 0
        preciesion_3 = 0
        preciesion_4 = 0
        preciesion_5 = 0
        count = 0
        filename_itor = getItor(anno_dir)
        for filename in filename_itor:
            print filename
            root = getRoot(filename)
            iou = self.test_image(root, consec=True)
            if iou > 0.0:
                preciesion_0 += 1
            if iou >= 0.1:
                preciesion_1 += 1
            if iou >= 0.2:
                preciesion_2 += 1
            if iou >= 0.3:
                preciesion_3 += 1
            if iou >= 0.4:
                preciesion_4 += 1
            if iou >= 0.5:
                preciesion_5 += 1
            count += 1
        #print preciesion_0
        print float(preciesion_0) / float(count)
        print float(preciesion_1) / float(count)
        print float(preciesion_2) / float(count)
        print float(preciesion_3) / float(count)
        print float(preciesion_4) / float(count)
        print float(preciesion_5) / float(count)
    
if __name__ == '__main__':
    #detector = CascadeDetector()
    #detector.detect_camera()
    #detector.release_camera()
    detector = CascadeDetector("image")
    detector.test()
    # detector.detect_image(test_dir)
    filename_itor = getItor(anno_dir)
    for filename in filename_itor:
        print filename
        root = getRoot(filename)
        detector.test_image(root)
    cv2.destroyAllWindows()
