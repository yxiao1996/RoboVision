"""
verision: alpha
update: 12-5-2017
author: Aramisbuildtoys
"""
import numpy as np
import cv2
import os
import glob 

def _detect_annotation(event,x,y,flags,param):
    global DOWN_X, DOWN_Y 
    global UP_X, UP_Y, DRAW_FLAG
    IMAGE = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        DOWN_X = x
        DOWN_Y = y
        DRAW_FLAG = True
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if DRAW_FLAG == True: 
            cv2.rectangle(IMAGE,(DOWN_X,DOWN_Y),(x,y),(0,255,0),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        cv2.rectangle(IMAGE,(DOWN_X,DOWN_Y),(x,y),(0,255,0),-1)
        DRAW_FLAG = False
        UP_X = x
        UP_Y = y
        
class CascadeTrainer():
    """using OpenCV to train cascade classifier in following order: 
       1. load and annotate training images
       2. make training data file
       3. train cascade classifier
       4. test classifier
    """
    def __init__(self, system="Windows"):
        """initilize training variables"""
        if system not in ["Ubuntu", "Windows"]:
            raise Exception("operating system should be Windows or Ubuntu!")
        self.pos_img = []
        self.neg_img = []
        self.pos_size = {}
        self.pos_anno = {}
        self.neg_size = {}
        self.neg_anno = {}
        self.params = []
        self.param_values = {}
        self.debug = True
        self.sys = system
        #self.show_data = "False"
        # the training procedure
        #self.load_image()
        #self.load_size()
        ##self.load_param()
        #self.write_negative()
        #self.write_positive()
        self.create_positive_samples()
        #self.check_positive_samples()
        self.train_classifier()

    def load_image(self):
        """loading images' file name"""
        print "loading image names..."
        # positive sample iterators
        pos_imgs_jpeg = glob.iglob(r'.\\pos\\*.jpeg')
        pos_imgs_jpg = glob.iglob(r'.\\pos\\*.jpg')
        # negative sample iterators
        neg_imgs_jpeg = glob.iglob(r'.\\neg\\*.jpeg')
        neg_imgs_jpg = glob.iglob(r'.\\neg\\*.jpg')
        # add file names to list
        for img in pos_imgs_jpeg:
            name = img.split('\\')[1:]
            name = '\\'.join(name[1:])
            self.pos_img.append(name)
        for img in pos_imgs_jpg:
            name = img.split('\\')[1:]
            name = '\\'.join(name[1:])
            self.pos_img.append(name)
        for img in neg_imgs_jpeg:
            self.neg_img.append(img)
        for img in neg_imgs_jpg:
            self.neg_img.append(img)
        print "number of positive samples: ", len(self.pos_img)
        print "number of negative samples: ", len(self.neg_img)
        # initialize dictionaries
        self.pos_size = {img: [] for img in self.pos_img}
        self.pos_anno = {img: [] for img in self.pos_img}
        self.neg_size = {img: [] for img in self.neg_img}
        self.neg_anno = {img: [] for img in self.neg_img}
        
        if self.debug:
            print "debug sample positive: ", self.pos_img[0]
            print "debug sample negative: ", self.neg_img[0]
        
    def load_size(self):
        """loading images' shape"""
        print "loading image shapes..."
        for img in self.pos_img:
            image = cv2.imread(img)
            shape = np.array(image).shape
            self.pos_size[img].append(shape)
        for img in self.neg_img:
            image = cv2.imread(img)
            shape = np.array(image).shape
            self.neg_size[img].append(shape)
        print "done."
        
        if self.debug:
            print "debug sample positive: ", self.pos_size[self.pos_img[0]]
            print "debug sample negative: ", self.neg_size[self.neg_img[0]]

    def load_param(self):
        """loading parameters for training"""
        print "loading training parameters..."
        fp = open('.\\params.txt', 'r')
        param_content = fp.readlines()
        for info in param_content:
            name = info.split(' ')[0]
            param = info.split(' ')[1].split('\n')[0]
            self.params.append(name)
            self.param_values[name] = param
            print name, " ", param 
        print "done."

    def write_negative(self):
        """writing background file"""
        print "writing background file..."
        fp = open('.\\bg.txt', 'w+')
        bg_content = fp.readlines()
        for img in self.neg_img:
            name = img.split('\\')[1:]
            name = '\\'.join(name[1:])
            fp.write(name)
            fp.write('\n')
        print "done."
        fp.close()
        
        if self.debug:
            fp = open('.\\bg.txt', 'r')
            bg_content = fp.readlines()
            print "debug sample: ", bg_content
            fp.close()
        
    def write_positive(self):
        """writing positive file"""
        print "loading annotation from positive file..."
        fp = open('.\\info.dat', 'a+')
        pos_content = fp.readlines()
        pos_img_prev = []
        pos_num_prev = {}
        pos_anno_prev = {}
        for info in pos_content:
            name = info.split(' ')[0]            
            num = info.split(' ')[1]
            anno = info.split(' ')[2:]
            pos_img_prev.append(name)
            pos_num_prev[name] = num
            pos_anno_prev[name] = anno
        for img in self.pos_img:
            if img not in pos_img_prev:
                print "new image found, have to add annotation"
                image = cv2.imread(img)
                IMAGE = image.copy()
                cv2.namedWindow('image')
                cv2.setMouseCallback('image',_detect_annotation, [IMAGE])
                print "press 'y' to confirm"
                while(True):
                    cv2.imshow('image',IMAGE)
                    if cv2.waitKey(20) & 0xFF== ord('y'):
                        break
                cv2.destroyAllWindows()
                new_anno = []
                new_anno.append(DOWN_X)
                new_anno.append(DOWN_Y)
                new_anno.append(UP_X)
                new_anno.append(UP_Y)
                fp.write(img)
                fp.write(' 1 ')
                for i in new_anno:
                    fp.write(str(i))
                    fp.write(' ')
                fp.write('\n')
        print "done."
        fp.close()
        
        if self.debug:
            fp = open('.\\info.dat', 'r')
            pos_content = fp.readlines()
            print "number of previous samples: ", len(pos_img_prev)
            print "number of current samples: ", len(pos_content)
            fp.close()
    
    def create_positive_samples(self):
        """using opencv_createsamples to generate positive samples"""
        print "generating positive data..."
        cmmd = str("opencv_createsamples"
                   +" -bg "+'./bg.txt'
                   +" -info "+'./info.dat' #self.param_values['-info']
                   +" -w "+"40"
                   +" -h "+"40"
                   +" -num "+"260"
                   +" -vec "+'./gen/vec/pos.vec')
        
        os.system(cmmd)
        print "done."

    def check_positive_samples(self):
        print "checking posotive data"
        cmmd = str("opencv_createsamples"
                   +" -vec './gen/pos.vec '"
                   +" -w "+self.param_values['-w']
                   +" -h "+self.param_values['-h'])
        os.system(cmmd)
        print "done."

    def train_classifier(self):
        """using opencv_haartraining to train classifier"""
        print "training classifier..."
        cmmd = str("opencv_traincascade"
                   +" -data "+'./gen'
                   +" -vec "+'./gen/vec/pos.vec'
                   +" -bg "+'./bg.txt'
                   +" -w "+"40"
                   +" -h "+"40"
                   +" -numPos "+"200"
                   +" -numNeg "+"410"
                   +" -numStages "+"19")
        print cmmd
        os.system(cmmd)
        print "done."

if __name__ == '__main__':
    train = CascadeTrainer()
