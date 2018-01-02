#!/usr/bin/python
from visual_odometry.Frame import *
from visual_odometry.Camera import *
from visual_odometry.util import *
from visual_odometry.Map import *
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
import cv2

class VisualOdometsy():
    def __init__(self):
        self.node_name = "visual_odometry"
        self.bridge = CvBridge()
        self.framerate = 10
        self.status = "init"
        self.ok = False
        self.map = Map()
        self.lossframe = 0
        # Subscribers
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.cbImage, queue_size=1)
        # Timers
        rospy.Timer(rospy.Duration.from_sec(1.0/self.framerate), self.mainLoop)

    def cbImage(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.cur_frame = Frame(cv_image)
        #self.cur_frame.plotFeature()
        self.ok = True
    
    def mainLoop(self, _event):
        if self.ok == False:
            return
        else:
            if self.status == "init":
                self.map.pushFrame(self.cur_frame)
                self.status = "default"
                return
            else:
                print len(self.cur_frame.keypoints)
                ref_frame = self.map.popFrame()
                try:
                    cur_frame = self.cur_frame
                    #ref_frame = self.map.popFrame()
                    matches = matchBF(cur_frame.descriptors, ref_frame.descriptors)
                    cur_kp, ref_kp = convertKeypoint(cur_frame.keypoints, ref_frame.keypoints, matches)
                    K = np.array(cur_frame.getK()).reshape([3, 3])
                    print len(cur_kp), len(ref_kp)
                    focal, pp = cur_frame.getFocalPp()
                    #print focal, pp
                    #E, _ = cv2.findEssentialMat(cur_kp, ref_kp, K, cv2.RANSAC, threshould=1.0)
                    E, _ = cv2.findEssentialMat(cur_kp, ref_kp, focal[0], pp, cv2.RANSAC, threshold=1.0)
                    _, R, t, mask = cv2.recoverPose(E, cur_kp, ref_kp, K, 1.0)

                    dR = R - np.eye(3, dtype=float)
                    d = np.sum(dR * dR)
                    print d
                    if d > 1:
                        #self.map.pushFrame(cur_frame)
                        raise Exception()
                    dt = t - np.array([0,0,0])
                    d = np.sum(dt * dt)
                    print d
                    if d > 5:
                        raise Exception()

                    pre_R, pre_t = ref_frame.getRt()
                    cur_R = np.dot(pre_R, R)
                    cur_t = pre_t + t
                    print cur_R
                    print cur_t
                    cur_frame.setRt(cur_R, cur_t)
                    self.lossframe = 0
                    self.map.pushFrame(cur_frame)
                except:
                    self.map.pushFrame(ref_frame)
                    self.lossframe = self.lossframe + 1
                    print "loss frame ", self.lossframe

    def plotFeature(self, _event):
        if self.status == "default":
            #print self.cur_frame.keypoints
            self.map.pushFrame(self.cur_frame)
            print self.map.popFrame().keypoints

if __name__ == '__main__':
    rospy.init_node('visual_odometry',anonymous=False)
    vo = VisualOdometsy()
    rospy.spin()