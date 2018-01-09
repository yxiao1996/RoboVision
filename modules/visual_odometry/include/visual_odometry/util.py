import cv2
import numpy as np
import drawMatches
from matplotlib import pyplot as plt

def detectORB(img):    
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    
    return kp, des

def matchBF(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = []
    min_dist = 10000
    max_dist = 0
    for i in range(len(matches)):
        if matches[i].distance < min_dist:
            min_dist = matches[i].distance
        if matches[i].distance > max_dist:
            max_dist = matches[i].distance

    for i in range(len(matches)):
        if matches[i].distance < max(2*min_dist, 30):
            good_matches.append(matches[i])

    sort_matches = sorted(good_matches, key = lambda x: x.distance)
    
    return sort_matches

def plot_matches(img1, kp1, img2, kp2, matches):    
    #img4 = drawMatches.drawMatches(img1, kp1, img2, kp2, matches)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches, np.array([]))
    # display image
    plt.gray()
    plt.imshow(img4)
    plt.show()

def convertKeypoint(keypoints1, keypoints2, matches):
    """turning cv.KeyPoint structure to numpt array"""
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(keypoints1[matches[i].queryIdx].pt)
        points2.append(keypoints2[matches[i].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)

    return points1, points2