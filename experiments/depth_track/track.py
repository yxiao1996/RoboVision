# -*- coding:utf-8 -*-
# author: Gene_ZC

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

camera_factor = 1000
camera_cx = 325.5
camera_cy = 253.5
camera_fx = 518.0
camera_fy = 519.0
sigma = 500

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

def depth_construction():
    filenames = os.listdir('depthexp/depth')
    fp = open('./dep.dat', 'a+')
    img_list = [cv2.imread(os.sep.join(['depthexp/depth', x]), -1) for x in filenames]
    with open('info.dat', 'r') as f:
        for img_iter in img_list:
            if img_iter is None:
                continue
            line = f.readline()
            _y, _x = int(line.split(' ')[0]), int(line.split(' ')[1])
            image = img_iter
            d = image[_x][_y]
            z = d / camera_factor
            x = (_y - camera_cx) * z / camera_fx + random.gauss(0, 2)
            y = (_x - camera_cy) * z / camera_fy + random.gauss(0, 0.125)
            fp.write(str(x) + ' ')
            fp.write(str(y) + ' ')
            fp.write(str(z) + ' ')
            fp.write('\n')
    fp.close()

def kalman_filter():
    pos = np.zeros((18, 2), dtype=np.float32)
    with open('dep.dat', 'r') as f:
        count = 0
        for line in f:
            _x, _y = line.split(' ')[0], line.split(' ')[1]
            pos[count][0] = _x
            pos[count][1] = _y
            count += 1
            if count > 18:
                break

    '''
    它有3个输入参数，dynam_params：状态空间的维数，这里为2；measure_param：测量值的维数，这里也为2; control_params：控制向量的维数，默认为0。由于这里该模型中并没有控制变量，因此也为0。
    '''
    kalman = cv2.KalmanFilter(2, 2)

    kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
    '''
    kalman.measurementNoiseCov为测量系统的协方差矩阵，方差越小，预测结果越接近测量值，kalman.processNoiseCov为模型系统的噪声，噪声越大，预测结果越不稳定，越容易接近模型系统预测值，且单步变化越大，相反，若噪声小，则预测结果与上个计算结果相差不大。
    '''

    kalman.statePre = np.array([[random.gauss(0, 0.1)], [random.gauss(0, 0.1)]], np.float32)

    f = open('kalman.dat', 'a+')
    for i in range(len(pos)):
        mes = np.reshape(pos[i, :], (2, 1))
        x = kalman.correct(mes)
        y = kalman.predict()
        f.write(str(y[0][0])+' ')
        f.write(str(y[1][0])+' ')
        f.write('\n')
    f.close()


def rgb_annotation():
    filenames = os.listdir('depthexp/rgb')
    fp = open('./info.dat', 'a+')
    img_list = [cv2.imread(os.sep.join(['depthexp/rgb', x])) for x in filenames]
    for img_iter in img_list:
        if img_iter is None:
            continue
        image = img_iter
        IMAGE = image.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', _detect_annotation, [IMAGE])
        print("press 'y' to confirm")
        while (True):
            cv2.imshow('image', IMAGE)
            if cv2.waitKey(20) & 0xFF == ord('y'):
                break
        cv2.destroyAllWindows()
        new_anno = []
        new_anno.append(DOWN_X)
        new_anno.append(DOWN_Y)
        new_anno.append(UP_X)
        new_anno.append(UP_Y)
        for i in new_anno:
            fp.write(str(i))
            fp.write(' ')
        fp.write('\n')
    fp.close()

def plot_3d():
    x = []
    y = []
    z = []
    with open('kalman.dat', 'r') as f:
        for line in f:
            _x, _y, _z = line.split(' ')[0], line.split(' ')[1], line.split(' ')[2]
            x.append(eval(_x))
            y.append(eval(_y))
            z.append(eval(_z))
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    plt.axis("equal")
    ax.view_init(elev=0, azim=90)
    ax.plot(x, y, z)
    # plt.show()
    fig.savefig('filter3.png', dpi=400)
    '''
    ax.view_init(elev=10., azim=11)
    ax.scatter(x, y, z, c='g')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    '''

if __name__ == '__main__':
    # rgb_annotation()
    # depth_construction()
    # kalman_filter()
    plot_3d()
