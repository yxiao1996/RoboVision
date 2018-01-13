<update (this file): 1/12/2018
# Cascade Detectors Training(Version Beta)

* To get faimilar with the pipeline of training cascade classifier in OpenCV, please visit:
```
https://docs.opencv.org/2.4.11/doc/user_guide/ug_traincascade.html
```

* ### Contents 
 1. using opencv_createsamples to create positive data from a set of annotated images;
 2. using opencv_traincascade to train classifier;

* ### Prerequisites
 1. Ubuntu on your computer (tested on Ubuntu 16.04)
 2. OpenCV >= 2.4 (tested on Version 3.3.1)
     
    you may download OpenCV from official site:
    ```
    https://opencv.org/releases.html
    ```
 3. labelImg (generate annotation)
    
    please clone and build it from GitHub:
    ```
    https://github.com/tzutalin/labelImg
    ```
 4. Training data (positive/negative)
* ###Usage
  Please following these steps in order to train the Cascade Detector:
  
  0. build the following directory for this project;
```    
    Detector   
    |--pos
	|   |--(positive images...)
	|--neg
	|   |--(negative images...)
	|--gen
	|   |--vec
	|   |   |--(positive data files)
	|   |--(results...)
	|--TrainCascade.py
	|--CascadeDetector.py
    |--params.txt
```
  1. Use labelImg to generate annotations for positive data;
  2. Use annoCvt.py to convert .xml annotation files to the OpenCV corresponding
     info.dat file;
  3. Use bgGen.py to generate background file bg.txt;
  4. Use TrainCascade.py to train classifier.
