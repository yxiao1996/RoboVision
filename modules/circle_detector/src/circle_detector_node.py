#!/usr/bin/python
from circle_detector.circle_detector import *
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
import cv2

class CircleDetectorNode():
    def __init__(self):
        self.node_name = "Circle_detector"
        self.bridge = CvBridge()
        self.framerate = 20
        self.detector = CircleDetector()
        self.status = "init"
        # Subscribers
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.cbImage, queue_size=1)
        # Timers
        rospy.Timer(rospy.Duration.from_sec(1.0/self.framerate), self.mainLoop)
    
    def cbImage(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.rbg = cv_image
        self.status = "default"

    def mainLoop(self, _event):
        if self.status == "default":
            self.detector.setImage(self.rbg)
            Circles_white, normals_white, centers_white, area_white = self.detector.detectCircles('white')
            self.detector.drawCircles(Circles_white, (255,0,0))
            self.detector.drawNormals(centers_white, normals_white, (255, 0, 0))
            img1 = self.detector.getImage()
            cv2.imshow('image', img1)
            cv2.waitKey(2)
        
if __name__ == '__main__':
    rospy.init_node('Circle_detector_node',anonymous=False)
    vo = CircleDetectorNode()
    rospy.spin()