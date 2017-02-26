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
        self.bgr_image = None                        # the latest image from the camera
        self.hsv_image = None
        self.filt_image = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        cv2.namedWindow('filt_window')

        # Create sliders to control image color filtering
        cv2.namedWindow('threshold_image')
        self.h_lower_bound = 25
        self.h_upper_bound = 38
        self.s_lower_bound = 199
        self.s_upper_bound = 255
        self.v_lower_bound = 203
        self.v_upper_bound = 237
        self.blur_amount = 3
        cv2.createTrackbar('H lower bound', 'threshold_image', 0, 255, self.set_h_lower_bound)
        cv2.createTrackbar('H upper bound', 'threshold_image', 0, 255, self.set_h_upper_bound)
        cv2.createTrackbar('S lower bound', 'threshold_image', 0, 255, self.set_s_lower_bound)
        cv2.createTrackbar('S upper bound', 'threshold_image', 0, 255, self.set_s_upper_bound)
        cv2.createTrackbar('V lower bound', 'threshold_image', 0, 255, self.set_v_lower_bound)
        cv2.createTrackbar('V upper bound', 'threshold_image', 0, 255, self.set_v_upper_bound)
        cv2.createTrackbar('Blur amount', 'threshold_image', 0, 20, self.set_blur_amount)
        cv2.setTrackbarPos('H lower bound', 'threshold_image', self.h_lower_bound)
        cv2.setTrackbarPos('H upper bound', 'threshold_image', self.h_upper_bound)
        cv2.setTrackbarPos('S lower bound', 'threshold_image', self.s_lower_bound)
        cv2.setTrackbarPos('S upper bound', 'threshold_image', self.s_upper_bound)
        cv2.setTrackbarPos('V lower bound', 'threshold_image', self.v_lower_bound)
        cv2.setTrackbarPos('V upper bound', 'threshold_image', self.v_upper_bound)
        cv2.setTrackbarPos('Blur amount', 'threshold_image', self.blur_amount)

        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def set_h_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue lower bound """
        self.h_lower_bound = val

    def set_h_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue upper bound """
        self.h_upper_bound = val

    def set_s_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation lower bound """
        self.s_lower_bound = val

    def set_s_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation upper bound """
        self.s_upper_bound = val

    def set_v_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the value lower bound """
        self.v_lower_bound = val

    def set_v_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the value upper bound """
        self.v_upper_bound = val

    def set_blur_amount(self, val):
        """ A callback function to handle the OpenCV slider to select the blur amount """
        # The kernel blur size must always be odd
        self.blur_amount = 2*val+1

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(cv2.GaussianBlur(self.bgr_image, \
            (self.blur_amount, self.blur_amount), 0), cv2.COLOR_BGR2HSV)
        self.filt_image = cv2.inRange(self.hsv_image, (self.h_lower_bound, self.s_lower_bound, \
            self.v_lower_bound), (self.h_upper_bound, self.s_upper_bound, self.v_upper_bound))
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.bgr_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.bgr_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # contours, hierarchy = cv2.findContours(self.filt_image, cv2.RETR_TREE, \
        #    cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(contours[0])
        x, y, w, h = 100, 100, 50, 50
        left_top = (x, y)
        right_bottom = (x+w, y+h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.bgr_image is None:
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.bgr_image)
                cv2.imshow('filt_window', self.filt_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
