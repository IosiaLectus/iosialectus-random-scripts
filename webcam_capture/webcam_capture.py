#!/usr/bin/python3
# program to capture single image from webcam in python
# Taken from https://www.geeksforgeeks.org/how-to-capture-a-image-from-webcam-in-python/, adjusted with light editing

import sys
import time
# importing OpenCV library
import cv2

# Number of photos to take and time interval between them
num_shots = int(sys.argv[1])
wait_time = float(sys.argv[2])

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cam_port = int(sys.argv[3])
cam = cv2.VideoCapture(cam_port)

# image capture loop
for i in range(num_shots):
    # reading the input using the camera
    result, image = cam.read()
    now = time.localtime(time.time())

    # If image will detected without any error,
    # show result
    if result:
        # saving image in local storage
        cv2.imwrite("photos/opencv_photo_{}{}{}_{}{}{}.png".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec), image)

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

    # Sleep between photos
    time.sleep(wait_time)
