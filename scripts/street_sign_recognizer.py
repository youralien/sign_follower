#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from template_matcher import TemplateMatcher

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')

        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV


        self.image = None                           # the latest image from the camera
        self.cv_image = None                        # gaussian blurred image
        self.hsv_image = None                       # HSV form of image
        self.binary_image = None                    # Binary form of image
        self.grid_cell = None                       # portion of an image

        self.density = 0                            # amount of white in a grid cell
        self.threshold = .1                         # amount of white needed in a grid cell to be part of sign

        #the thresholds to find the yellow color of the sign
        self.hsv_lb = np.array([17, 161, 160]) # hsv lower bound
        self.hsv_ub = np.array([75, 255, 255]) # hsv upper bound

        # the various windows
        cv2.namedWindow('HSV_window')
        cv2.namedWindow('Binary_window')
        cv2.namedWindow('RGB_window')



        images = {
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'
        }

        self.matcher = TemplateMatcher(images)
        self.matched_threshold = 10
        self.total_confidence = {'uturn': 0.0, 'left': 0.0, 'right':0.0}
        self.recognized = False

        rospy.Subscriber("/camera/image_raw", Image, self.process_image)


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.medianBlur(self.image,5)

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.image, left_top, right_bottom, color=(0, 0, 255), thickness=5)


        moments = cv2.moments(self.binary_image)
        if moments['m00'] != 0:
            self.center_x, self.center_y = moments['m10']/moments['m00'], moments['m01']/moments['m00']

        self.imagesum = np.sum(self.binary_image)
        self.imagesum /= (255.0)
        self.imagesum /= (640.0)
        self.imagesum *= 10


        cv2.circle(self.image, (int(self.center_x), int(self.center_y)), int(self.imagesum), (128, 255, 128), thickness=-1)

        detected_sign = self.image[top:bottom, left:right]
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


        if not self.recognized and detected_sign != []:
            sign_confidences = self.matcher.predict(gray_image)
            for sign, confidence in sign_confidences.items():
                self.total_confidence[sign] += confidence

            for sign, tot_confidence in self.total_confidence.items():
                if tot_confidence > self.matched_threshold:
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

        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.hsv_image, (self.hsv_lb[0], self.hsv_lb[1],self.hsv_lb[2]), (self.hsv_ub[0],self.hsv_ub[1],self.hsv_ub[2]))
        left_top = (640, 480)
        right_bottom = (0, 0)
        for r in range(1,11):
            for c in range(1, 11):
                grid_cell_x = 64*(c-1)
                grid_cell_y = 48*(r-1)
                grid_cell = self.binary_image[grid_cell_y:grid_cell_y+48, grid_cell_x:grid_cell_x+64]
                self.density = np.sum(grid_cell)
                self.density /= (255.0*48*64)
                if(self.density > self.threshold):
                    if(grid_cell_x < left_top[0] or grid_cell_y < left_top[1]):
                        left_top = (grid_cell_x, grid_cell_y)
                    if((grid_cell_x+64) > right_bottom[0] or (grid_cell_y+48) > right_bottom[1]):
                        right_bottom = (grid_cell_x+64, grid_cell_y+48)


        if(right_bottom == (0,0)):
            left_top = (320, 240)
            right_bottom = (320, 240)


        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.image is None:
                # creates a window and displays the image for X milliseconds
                cv2.imshow('HSV_window', self.hsv_image)
                cv2.imshow('Binary_window', self.binary_image)
                cv2.imshow('RGB_window', self.image)

                cv2.waitKey(5)
            r.sleep()


if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
