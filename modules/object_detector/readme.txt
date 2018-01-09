# Training Cascade Detectors (Version Alpha)

to get faimilar with the pipeline of training cascade classifier in OpenCV, please visit:
    https://docs.opencv.org/2.4.11/doc/user_guide/ug_traincascade.html

this wrapper using the following configureations:
1. using opencv_createsamples to create positive data from a set of annotated images;
2. image annotation process is integrated in the training process (for now, only one annotation is allowed);
3. using opencv_traincascade to train classifier;	
	
to train and test the classifier you have to follow these precedures:
0. install Python, OpenCV(Python API) and numpy on your computer;
1. build directories in a proper struture;
2. collect positive image and negative image and put them in correct place;
3. edit training parameters (params.txt);
4. run python script to train classifier (python TrainCascade.py);
   4.1. add annotations for your new images;
5. run python script to test classifier (python CascadeDetector.py);
6. selectivly go back to step 2

here are some more notes:
1. directories should have the following struture:
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
2. in order to run the test script, your computer must have (or connected to) a camera (webcam etc.)
3. annotations must done in this fashion: left button of mouse down at left upper corner of object, drag to right lower corner then release
4. more training parameters may be added to the script
