#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

from __future__ import division
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
        self.cv_bgr_image = None                    # the latest image from the camera
        self.cv_hsv_image = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')

        self.template_matcher = TemplateMatcher()

        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        self.image_info_window = None
        self.hsv_lb = np.array([20, 170, 154])  # hsv lower bound
        self.hsv_ub = np.array([33, 255, 255])  # hsv upper bound

        # morphology setup
        kernel_size = 5
        self.morphology_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self.bounding_box_extend = 30

        # tool to find HSV in image (TODO: refactor into new node)
        self.get_hue_range_tool = False
        if self.get_hue_range_tool:
            cv2.namedWindow('threshold_image')
            cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
            cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
            cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
            cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
            cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
            cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)

    def set_h_lb(self, val):
        """ set hue lower bound """
        self.hsv_lb[0] = val

    def set_s_lb(self, val):
        """ set saturation lower bound """
        self.hsv_lb[1] = val

    def set_v_lb(self, val):
        """ set value lower bound """
        self.hsv_lb[2] = val

    def set_h_ub(self, val):
        """ set hue upper bound """
        self.hsv_ub[0] = val

    def set_s_ub(self, val):
        """ set saturation upper bound """
        self.hsv_ub[1] = val

    def set_v_ub(self, val):
        """ set value upper bound """
        self.hsv_ub[2] = val

    def image_callback(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_bgr_image for subsequent processing """
        self.cv_bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_hsv_image = cv2.cvtColor(self.cv_bgr_image, cv2.COLOR_BGR2HSV)
        self.image_processed = False

    def find_yellow_parts(self, hsv_image):
        """
        Takes: HSV image
        Returns a thresholded image that contains the yellow parts of the scene
        """
        threshold_image = cv2.inRange(
            hsv_image,
            self.hsv_lb,
            self.hsv_ub
        )

        return threshold_image

    def find_big_parts(self, threshold_image):
        """
        Takes: Grayscale threshold image
        Returns a threshold image with smaller parts removed
        """
        # use morphology to get rid of small parts
        morphed = cv2.erode(threshold_image, self.morphology_kernel, iterations=1)
        # and make everything we found a bit bigger (probably unnecessary)
        morphed = cv2.dilate(morphed, self.morphology_kernel, iterations=1)

        return morphed

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        yellow_threshold = self.find_yellow_parts(self.cv_hsv_image)
        big_yellow_parts = self.find_big_parts(yellow_threshold)

        max_width = self.cv_hsv_image.shape[1] - 1
        max_height = self.cv_hsv_image.shape[0] - 1

        x, y, w, h = cv2.boundingRect(big_yellow_parts)
        r = self.bounding_box_extend
        left_top = (
            max(x - r, 0),
            max(y - r, 0)
        )

        right_bottom = (
            min(x + w + r, max_width),
            min(y + h + r, max_height)
        )

        return left_top, right_bottom

    def recognize_sign(self):
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_bgr_image[top:bottom, left:right]

        # attempt to recognize sign
        sign_type_distribution = self.template_matcher.predict(
            cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
        )

        most_likely_sign_type = max(sign_type_distribution, key=sign_type_distribution.get)
        print most_likely_sign_type
        if sign_type_distribution[most_likely_sign_type] > 0.5:
            cv2.putText(
                self.cv_bgr_image,
                "{}: {}".format(most_likely_sign_type, sign_type_distribution[most_likely_sign_type]),
                left_top,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255)
            )

        # draw bounding box rectangle
        cv2.rectangle(self.cv_bgr_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.cv_bgr_image is not None:
                # prevent reprocessing images
                if not self.image_processed:
                    self.recognize_sign()
                    self.image_processed = True

                # creates a window and displays the image for X milliseconds
                cv2.imshow(
                    'video_window',
                    self.cv_bgr_image
                )
                cv2.waitKey(5)

            if self.image_info_window is not None and self.get_hue_range_tool:
                cv2.imshow('image_info', self.image_info_window)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
