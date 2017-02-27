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
        self.cv_image = None  # the latest image from the camera
        self.hsv_img = None
        self.binary_img = None
        self.bridge = CvBridge()  # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        cv2.namedWindow('HSV_window')
        cv2.namedWindow('binary_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

        self.hsv_min = (20, 200, 200)
        self.hsv_max = (40, 255, 255)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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
        # TODO: YOUR SOLUTION HERE
        self.hsv_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

        self.binary_img = cv2.inRange(self.hsv_img, self.hsv_min, self.hsv_max)

        left_top = (200, 200)
        right_bottom = (400, 400)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('HSV_window', self.hsv_img)
                cv2.imshow('binary_window', self.binary_img)
                cv2.waitKey(5)
            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print "detected timeskip"


if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
