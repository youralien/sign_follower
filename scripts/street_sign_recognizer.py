#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy, cv2
import cv2_utils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import template_matcher as tm

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """

    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

        self.cv_image = None                        # the latest image from the camera
        self.hsv_img = None                         # the image in hsv colorspace
        self.binary_img = None                      # the filtered image
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        # self.tm = tm.TemplateMatcher()

        cv2.namedWindow('raw')
        cv2.namedWindow('hsv')
        cv2.namedWindow('bin')

        # Configuration (eg. hsv bounds) are loaded in using a custom config object
        # The sliders are generated automatically by that config object as well and
        # any new values are saved automatically at shutdown
        cfgFilepath = rospy.get_param('~cfgFilepath')
        self.config = cv2_utils.Config(cfgFilepath)
        self.config.sliders('hsv')
        rospy.on_shutdown(self.config.save)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        left_top, right_bottom = self.sign_bounding_box(self.cv_image)
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # cropped_grayscale = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('raw', cropped_grayscale)
        # cv2.waitKey()
        # self.tm.predict(cropped_grayscale)

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def sign_bounding_box(self, raw_img):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining top-left and bottom-right corners of the bounding box
        """

        self.hsv_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

        self.binary_img = cv2.inRange(
            self.hsv_img,
            tuple(self.config.get(["lb"], ["h", "s", "v"])),
            tuple(self.config.get(["ub"], ["h", "s", "v"])),
        )

        contours, heirarchy = cv2.findContours(
            np.copy(self.binary_img),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_TC89_L1
        )

        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(cnt)

        left_top = (x, y)
        right_bottom = (x+w, y+h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():

            # creates a window and displays the image for X milliseconds
            if self.cv_image is not None:
                cv2.imshow('raw', self.cv_image)
            if self.hsv_img is not None:
                cv2.imshow('hsv', self.hsv_img)
            if self.binary_img is not None:
                cv2.imshow('bin', self.binary_img)

            cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
