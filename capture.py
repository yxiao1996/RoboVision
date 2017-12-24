# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import time
from std_msgs.msg import String
from VO import VisualOdometry
# Instantiate CvBridge
bridge = CvBridge()

STAGE_INI = 0
STAGE_DEF = 1
img_buf = './tmp/tmp.jpeg'
face_cascade = cv2.CascadeClassifier("/home/robocon/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml")

translation = [0, 0, 0]

def detect_face(face_cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    print len(faces)
    img = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
    cv2.imshow('img', frame)
    return img

def image_callback(msg):
    global translation
    #print("Received an image!")
    # face lib
    # face_cascade = cv2.CascadeClassifier("/home/robocon/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml")
    vo = VisualOdometry(debug=False)
    try:
        # Convert your ROS Image message to OpenCV2
        cur_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
	stage = rospy.get_param('stage')
        if stage == STAGE_INI:
	    # save first image in tmp         
	    cv2.imwrite(img_buf, cur_img)
            rospy.set_param('stage', STAGE_DEF)
        else:
            # load image from tmp
            pre_img = cv2.imread(img_buf)
            R1, R2, t1, t2 = vo.calcu_R_t(cur_img, pre_img)
            for i in range(len(translation)):
                translation[i] += t1[i]
            print translation
            # pre_t = rospy.get_param('translation')
            # cur_t = pre_t + t1
            # print cur_t
            # rospy.set_param('translation', cur_t)
            # save current image in tmp
            cv2.imwrite(img_buf, cur_img)

def main():    
    #init node
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/kinect2/qhd/image_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # index parameter
    rospy.set_param('img_id', 0)
    rospy.set_param('stage', STAGE_INI)
    rospy.set_param('translation', [0, 0, 0])
    # Spin until ctrl + c
    translation = [0, 0, 0]
    rospy.spin()

if __name__ == '__main__':
    main()
