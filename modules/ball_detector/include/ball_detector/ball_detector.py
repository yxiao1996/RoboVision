import numpy as np
import cv2
from matplotlib import pyplot as plt

class LineDetector():
    def __init__(self):
        # Images to be processed
        self.bgr = np.empty(0)
        self.hsv = np.empty(0)
        self.edges = np.empty(0)

        # Color value range in HSV space: default
        self.hsv_white1 = np.array([0, 0, 150])
        self.hsv_white2 = np.array([180, 60, 255])
        self.hsv_yellow1 = np.array([25, 140, 100])
        self.hsv_yellow2 = np.array([45, 255, 255])
        self.hsv_red1 = np.array([0, 140, 100])
        self.hsv_red2 = np.array([10, 255, 255])
        self.hsv_red3 = np.array([170, 140, 100])
        self.hsv_red4 = np.array([180, 255, 255])
        self.hsv_green1 = np.array([41, 40, 40])
        self.hsv_green2 = np.array([90, 255, 255])
        self.hsv_orange1 = np.array([20, 140, 100])
        self.hsv_orange2 = np.array([35, 255, 255])
        self.hsv_pink1 = np.array([165, 40, 40])
        self.hsv_pink2 = np.array([180, 255, 255])
        self.hsv_blue1 = np.array([90, 140, 100])
        self.hsv_blue2 = np.array([140, 255, 255])

        # Parameters for dilation, Canny, and etc: default
        self.dilation_kernel_size = 3
        self.canny_thresholds = [80,200]
        self.hough_threshold  = 20
        self.hough_min_line_length = 3
        self.hough_max_line_gap = 1
        self.max_color_dist=150
        self.coeff_yellow=1.25
        self.coeff_white=0.75
        self.sobel_threshold = 40.

    def _colorFilter(self, color):
        # threshold colors in HSV space
        if color == 'white':
            bw = cv2.inRange(self.hsv, self.hsv_white1, self.hsv_white2)
        elif color == 'yellow':
            bw = cv2.inRange(self.hsv, self.hsv_yellow1, self.hsv_yellow2)
        elif color == 'green':
            bw = cv2.inRange(self.hsv, self.hsv_green1, self.hsv_green2)
        elif color == 'orange':
            bw = cv2.inRange(self.hsv, self.hsv_orange1, self.hsv_orange2)
        elif color == 'pink':
            bw = cv2.inRange(self.hsv, self.hsv_pink1, self.hsv_pink2)
        elif color == 'blue':
            bw = cv2.inRange(self.hsv, self.hsv_blue1, self.hsv_blue2)
        elif color == 'red':
            bw1 = cv2.inRange(self.hsv, self.hsv_red1, self.hsv_red2)
            bw2 = cv2.inRange(self.hsv, self.hsv_red3, self.hsv_red4)
            bw = cv2.bitwise_or(bw1, bw2)
        else:
            raise Exception('Error: Undefined color strings...')

        # binary dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.dilation_kernel_size, self.dilation_kernel_size))
        
        # refine edge for certain color
        edge_color = cv2.bitwise_and(cv2.dilate(bw, kernel), self.edges)

        return bw, edge_color

    def _lineFilter(self, bw, edge_color):
        # find gradient of the bw image
        grad_x = -cv2.Sobel(bw/255, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = -cv2.Sobel(bw/255, cv2.CV_32F, 0, 1, ksize=5)
        grad_x *= (edge_color == 255)
        grad_y *= (edge_color == 255)

        # compute gradient and thresholding
        grad = np.sqrt(grad_x**2 + grad_y**2)
        roi = (grad>self.sobel_threshold)

        #print np.unique(grad)
        #print np.sum(roi)

        # turn into a list of points and normals
        roi_y, roi_x = np.nonzero(roi)
        centers = np.vstack((roi_x, roi_y)).transpose()
        normals = np.vstack((grad_x[roi], grad_y[roi])).transpose()
        normals /= np.sqrt(np.sum(normals**2, axis=1, keepdims=True))

        lines = self._synthesizeLines(centers, normals)

        return lines, normals, centers

    def _correctPixelOrdering(self, lines, normals):
        flag = ((lines[:,2]-lines[:,0])*normals[:,1] - (lines[:,3]-lines[:,1])*normals[:,0])>0
        for i in range(len(lines)):
            if flag[i]:
                x1,y1,x2,y2 = lines[i, :]
                lines[i, :] = [x2,y2,x1,y1] 

    def _findNormal(self, bw, lines):
        normals = []
        centers = []
        if len(lines)>0:
            length = np.sum((lines[:, 0:2] -lines[:, 2:4])**2, axis=1, keepdims=True)**0.5
            dx = 1.* (lines[:,3:4]-lines[:,1:2])/length
            dy = 1.* (lines[:,0:1]-lines[:,2:3])/length

            centers = np.hstack([(lines[:,0:1]+lines[:,2:3])/2, (lines[:,1:2]+lines[:,3:4])/2])
            x3 = (centers[:,0:1] - 3.*dx).astype('int')
            y3 = (centers[:,1:2] - 3.*dy).astype('int')
            x4 = (centers[:,0:1] + 3.*dx).astype('int')
            y4 = (centers[:,1:2] + 3.*dy).astype('int')
            x3 = self._checkBounds(x3, bw.shape[1])
            y3 = self._checkBounds(y3, bw.shape[0])
            x4 = self._checkBounds(x4, bw.shape[1])
            y4 = self._checkBounds(y4, bw.shape[0])
            flag_signs = (np.logical_and(bw[y3,x3]>0, bw[y4,x4]==0)).astype('int')*2-1
            normals = np.hstack([dx, dy]) * flag_signs
 
            """ # Old code with lists and loop, performs 4x slower 
            for cnt,line in enumerate(lines):
                x1,y1,x2,y2 = line
                dx = 1.*(y2-y1)/((x1-x2)**2+(y1-y2)**2)**0.5
                dy = 1.*(x1-x2)/((x1-x2)**2+(y1-y2)**2)**0.5
                x3 = int((x1+x2)/2. - 3.*dx)
                y3 = int((y1+y2)/2. - 3.*dy)
                x4 = int((x1+x2)/2. + 3.*dx)
                y4 = int((y1+y2)/2. + 3.*dy)
                x3 = self._checkBounds(x3, bw.shape[1])
                y3 = self._checkBounds(y3, bw.shape[0])
                x4 = self._checkBounds(x4, bw.shape[1])
                y4 = self._checkBounds(y4, bw.shape[0])
                if bw[y3,x3]>0 and bw[y4,x4]==0:
                    normals[cnt,:] = [dx, dy] 
                else:
                    normals[cnt,:] = [-dx, -dy]
            """
            self._correctPixelOrdering(lines, normals)
        return centers, normals

    def _findEdge(self, gray):
        edges = cv2.Canny(gray, self.canny_thresholds[0], self.canny_thresholds[1], apertureSize = 3)
        return edges

    def _HoughLine(self, edge):
        lines = cv2.HoughLinesP(edge, 1, np.pi/180, self.hough_threshold, np.empty(1), self.hough_min_line_length, self.hough_max_line_gap)
        if lines is not None:
            lines = np.array(lines[0])
        else:
            lines = []
        return lines

    def _HoughCircle(self, img):
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,2,20,
                                   param1=200,param2=110,minRadius=0,maxRadius=0)
        if circles is not None:
            circles = np.int16(np.around(circles))
        else:
            circles = []
        return circles

    def _checkBounds(self, val, bound):
        val[val<0]=0
        val[val>=bound]=bound-1
        return val

    def _synthesizeLines(self, centers, normals):
        lines = []
        if len(centers)>0:
            x1 = (centers[:,0:1] + normals[:, 1:2] * 6.).astype('int')
            y1 = (centers[:,1:2] - normals[:, 0:1] * 6.).astype('int')
            x2 = (centers[:,0:1] - normals[:, 1:2] * 6.).astype('int')
            y2 = (centers[:,1:2] + normals[:, 0:1] * 6.).astype('int')
            x1 = self._checkBounds(x1, self.bgr.shape[1])
            y1 = self._checkBounds(y1, self.bgr.shape[0])
            x2 = self._checkBounds(x2, self.bgr.shape[1])
            y2 = self._checkBounds(y2, self.bgr.shape[0])
            lines = np.hstack([x1, y1, x2, y2])
        return lines

    def detectLines(self, color):
        bw, edge_color = self._colorFilter(color)
        lines = self._HoughLine(edge_color)
        #lines, normals, centers = self._lineFilter(bw, edge_color)
        #return Detections(lines=lines, normals=normals, area=bw, centers=centers)
        
        centers, normals = self._findNormal(bw, lines)
        return lines, normals, centers, bw

    def detectCircles(self):
        self.bgr = cv2.medianBlur(self.bgr,5)
        bw, edge_color = self._colorFilter('white')
        cv2.imshow("be", bw)
        cv2.waitKey(1)
        circles = self._HoughCircle(edge_color)
        
        self.drawCircles(circles)
        cv2.imshow("test", self.bgr)
        cv2.waitKey(0)

    def detectBall(self):
        bw_red, _ = self._colorFilter('red')
        bw_yellow, _ = self._colorFilter('orange')
        bw_green, _ = self._colorFilter('green')
        bw_blue, _ = self._colorFilter('blue')
        bw_or = cv2.bitwise_or(cv2.bitwise_or(bw_red, cv2.bitwise_or(bw_green, bw_yellow)), bw_blue)
        bw_or = cv2.medianBlur(bw_or,5)
        cv2.imshow('red', bw_red)
        cv2.imshow('yellw', bw_yellow)
        cv2.imshow('green', bw_green)
        cv2.imshow('blue', bw_blue)
        cv2.imshow('or', bw_or)
        circles = self._HoughCircle(bw_or)
                
        self.drawCircles(circles)
        cv2.imshow("test", self.bgr)
        cv2.waitKey(0)
        cv2.waitKey(0)

    def setImage(self, bgr):
        self.bgr = np.copy(bgr)
        self.hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.edges = self._findEdge(self.bgr)

    def getImage(self):
        return self.bgr

    def drawCircles(self, circles):
        if len(circles) > 0:
            print len(circles[0])
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(self.bgr,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(self.bgr,(i[0],i[1]),2,(0,0,255),3)

    def drawLines(self, lines, paint):
        if len(lines)>0:
            for x1,y1,x2,y2 in lines:
                cv2.line(self.bgr, (x1,y1), (x2,y2), paint, 2)
                cv2.circle(self.bgr, (x1,y1), 2, (0,255,0))
                cv2.circle(self.bgr, (x2,y2), 2, (0,0,255))

    def drawNormals(self, centers, normals, paint):
        if len(centers)>0:
            for x,y,dx,dy in np.hstack((centers,normals)):
                x3 = int(x - 2.*dx)
                y3 = int(y - 2.*dy)
                x4 = int(x + 2.*dx)
                y4 = int(y + 2.*dy)
                cv2.line(self.bgr, (x3,y3), (x4,y4), paint, 1)
                cv2.circle(self.bgr, (x3,y3), 1, (0,255,0))
                cv2.circle(self.bgr, (x4,y4), 1, (0,0,255))

def _main():
    img = cv2.imread('./img/0.jpeg')
    #d = LineDetector()
    d = LineDetector()
    d.setImage(img)
    d.detectBall()
    #lines_white, normals_white, centers_white, area_white = d.detectLines('white')
    #d.drawLines(lines_white, (0,255,0))
    #d.drawNormals(centers_white, normals_white, (255, 0, 0))
    #img1 = d.getImage()
    #while(1):
    #    cv2.imshow('img',np.array(img1))
    #plt.gray()
    #plt.imshow(img1)
    #plt.show()
    #print edge_color
if __name__ == '__main__':
    _main()