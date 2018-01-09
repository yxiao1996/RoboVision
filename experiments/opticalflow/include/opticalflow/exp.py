import numpy as np
import cv2

cap = cv2.VideoCapture(1)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Camera parameters
ret, f = cap.read()
img_w = np.array(f).shape[1]
img_h = np.array(f).shape[0]
print "image width: ", img_w, "image height: ", img_h

# ROI
roi = np.zeros_like(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
roi[img_h/2:, img_w/2:] = 1
print "roi shape: ", roi.shape
# Wait for experiment
print "press 't' to start tracking"
while(1):
    ret, f = cap.read()
    # draw ROI
    mask = np.zeros_like(f)
    mask = cv2.line(mask, (img_w/2,img_h/2), (img_w/2,img_h), color[0].tolist(), 2)
    mask = cv2.line(mask, (img_w/2,img_h/2), (img_w,img_h/2), color[0].tolist(), 2)
    mask = cv2.line(mask, (img_w,img_h/2), (img_w,img_h), color[0].tolist(), 2)
    mask = cv2.line(mask, (img_w/2,img_h), (img_w,img_h), color[0].tolist(), 2)
    # display
    img = cv2.add(f, mask)
    #img = np.multiply(f, roi)
    cv2.imshow('frame', img)
    
    k = cv2.waitKey(30) & 0xff
    if k == ord("t"):
        break

# Take first frame and find corners
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi, **feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord("q"):
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()