#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        cv2.namedWindow('binary_window')

        self.hue_lower_bound = 0
        self.hue_upper_bound = 50
        self.saturation_lower_bound = 175
        self.saturation_upper_bound = 255
        cv2.createTrackbar('hue lower bound', 'binary_window', 0, 255, self.set_hue_lower_bound)
        cv2.createTrackbar('hue upper bound', 'binary_window', 0, 255, self.set_hue_upper_bound)
        cv2.createTrackbar('saturation lower bound', 'binary_window', 0, 255, self.set_saturation_lower_bound)
        cv2.createTrackbar('saturation upper bound', 'binary_window', 0, 255, self.set_saturation_upper_bound)

    def set_hue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.hue_lower_bound = val

    def set_hue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.hue_upper_bound = val

    def set_saturation_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.saturation_lower_bound = val

    def set_saturation_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.saturation_upper_bound = val

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.cv_image,
            (self.hue_lower_bound,self.saturation_lower_bound,100),
            (self.hue_upper_bound,self.saturation_upper_bound,255))

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # _, contours, hierarchy = cv2.findContours(self.binary_image, 1, 2)
        #image_copy = self.binary_image
        points = np.transpose(np.nonzero(self.binary_image))
        x,y,w,h = cv2.boundingRect(points)
        left_top = (y, x)
        right_bottom = (y+h, x+w)
        # left_top = (200, 400)
        # right_bottom = (400, 600)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('hsv_window', self.hsv_image)
                cv2.imshow('binary_window', self.binary_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
