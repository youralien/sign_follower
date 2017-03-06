#!/usr/bin/env python
"""
To run street_sign_recognizer.py, please run:

roscd sign_follower
rosbag play (uturn.bag, rightturn.bag, or leftturn.bag) -l

rosrun image_transport republish compressed in:=/camera/image_raw _image_transport:=compressed raw out:=/camera/image_raw
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from template_matcher import TemplateMatcher

"""
CompRobo Spring 2017

Completed by Kevin Zhang

This script detects yellow pedestrian signs in its field of view and categorizes
them as either U-turn signs, Right turn signs, or Left turn signs. Steps involved:

1. Upon receiving an image, convert to a binary image that searchs for yellow color
2. Use grid cells to generate a bounding box on binary image
3. Select the bounding box and send it to Template Matcher for categorization
4. Output detected sign and recognition of specific sign
"""

class StreetSignRecognizer(object):
    """
    The main class of this script, which holds the sign detection methods and variables
    It outputs a detected sign which it sends to the Template Matcher module for
    categorization
    """
    def __init__(self):
        #init ROS node
        rospy.init_node('street_sign_recognizer')

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.image = None                           # the latest image from the camera
        self.cv_image = None                        # gaussian blurred image
        self.hsv_image = None                       # HSV form of image
        self.binary_image = None                    # Binary form of image
        self.grid_cell = None                       # portion of an image

        self.density = 0                            # amount of white in a grid cell
        self.threshold = .001                       # amount of white needed in a grid cell to be part of sign

        #the thresholds to find the yellow color of the sign
        self.hsv_lb = np.array([17, 161, 160]) # hsv lower bound
        self.hsv_ub = np.array([75, 255, 255]) # hsv upper bound

        # the various windows for visualization
        cv2.namedWindow('HSV_window')
        cv2.namedWindow('Binary_window')
        cv2.namedWindow('RGB_window')

        #set of template images for the Template Matcher
        images = {
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'
        }

        #variables for Template Matcher
        self.matcher = TemplateMatcher(images)                              # initialize Template Matcher class
        self.matched_threshold = 40                                         # threshold to determine which sign the input is
        self.total_confidence = {'uturn': 0.0, 'left': 0.0, 'right':0.0}    # dictionary that holds cumulative confidence of each sign
        self.recognized = False                                             #boolean to ensure only one run of the recognition

        #init ROS Subscriber to camera image
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


    def process_image(self, msg):
        """
        Process image messages from ROS by detecting a sign, and sends the sign
        to TemplateMatcher for categorization
        """
        #converts ROS image to OpenCV image
        self.image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #blurs image to average out some color
        self.cv_image = cv2.medianBlur(self.image,5)

        #finds the corners of the bounding box on the sign
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # draw bounding box rectangle
        cv2.rectangle(self.image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

        #crops the image to the detected sign given by the bounding box ranges
        detected_sign = self.image[top:bottom, left:right]
        #converts to grayscale for processing
        gray_image = cv2.cvtColor(detected_sign, cv2.COLOR_BGR2GRAY)

        #as long as detected_sign exists and a sign has not been recognized yet
        if not self.recognized and detected_sign != []:
            #get the dictionary of confidence values from TemplateMatcher for this image
            sign_confidences = self.matcher.predict(gray_image)
            #update the cumulative dictionary of confidence values
            for sign, confidence in sign_confidences.items():
                self.total_confidence[sign] += confidence

            #if a value in the cumulative dictionary passes the threshold
            for sign, tot_confidence in self.total_confidence.items():
                if tot_confidence > self.matched_threshold:
                    #declare that sign, and discontinue recognition
                    print "I know what this is! It's a", sign, "sign!"
                    self.recognized = True
                    break


    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        #converts RGB image to hue, saturation, value image
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        #creates a binary image on the basis of the yellow sign
        self.binary_image = cv2.inRange(self.hsv_image, (self.hsv_lb[0], self.hsv_lb[1],self.hsv_lb[2]), (self.hsv_ub[0],self.hsv_ub[1],self.hsv_ub[2]))

        left_top = (640, 480)
        right_bottom = (0, 0)
        #splits the image into 100 grid cells
        for r in range(1,11):
            for c in range(1, 11):
                grid_cell_x = 64*(c-1)
                grid_cell_y = 48*(r-1)
                grid_cell = self.binary_image[grid_cell_y:grid_cell_y+48, grid_cell_x:grid_cell_x+64]
                self.density = np.sum(grid_cell)
                self.density /= (255.0*48*64)
                #checks if current cell contains enough white to be considered part of the sign
                if(self.density > self.threshold):
                    #if so, check if current cell increases the width of height of the bounding box
                    if(grid_cell_x < left_top[0] or grid_cell_y < left_top[1]):
                        left_top = (grid_cell_x, grid_cell_y)
                    if((grid_cell_x+64) > right_bottom[0] or (grid_cell_y+48) > right_bottom[1]):
                        right_bottom = (grid_cell_x+64, grid_cell_y+48)

        #if nothing was found, then just create a point at the image's center
        if(right_bottom == (0,0)):
            left_top = (320, 240)
            right_bottom = (320, 240)

        return left_top, right_bottom


    def run(self):
        """
        The main run loop
        """
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.image is None:
                # creates the windows and displays the RGB, HSV, and binary image
                cv2.imshow('HSV_window', self.hsv_image)
                cv2.imshow('Binary_window', self.binary_image)
                cv2.imshow('RGB_window', self.image)

                cv2.waitKey(5)
            r.sleep()



if __name__ == '__main__':
    #initializes the Sign Recognition class and runs it
    node = StreetSignRecognizer()
    node.run()
