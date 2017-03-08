#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

from __future__ import division
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
        self.cv_bgr_image = None                    # the latest image from the camera
        self.cv_hsv_image = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

        self.image_info_window = None
        self.hsv_lb = np.array([20, 170, 154])  # hsv lower bound
        self.hsv_ub = np.array([33, 255, 255])  # hsv upper bound

        # morphology setup
        kernel_size = 5
        self.morphology_kernel = np.ones((kernel_size, kernel_size), np.uint8)

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

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_bgr_image for subsequent processing """
        self.cv_bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_hsv_image = cv2.cvtColor(self.cv_bgr_image, cv2.COLOR_BGR2HSV)

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_bgr_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_bgr_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def find_blobs(self, grayscale_image):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 10000

        # filter by area
        params.filterByArea = True
        params.minArea = 40  # units?

        # filter by circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # filter by convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # filter by inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(grayscale_image)

        print keypoints

        return cv2.drawKeypoints(grayscale_image, keypoints, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def find_yellow_parts(self, hsv_image):
        threshold_image = cv2.inRange(
            hsv_image,
            self.hsv_lb,
            self.hsv_ub
        )

        return threshold_image  # return cv2.filter2D(threshold_image, -1, kernel)

    def find_big_parts(self, threshold_image):
        # use morphology to get rid of small parts
        eroded = cv2.erode(threshold_image, self.morphology_kernel, iterations=1)
        # and make everything we found a bit bigger
        dilated = cv2.dilate(eroded, self.morphology_kernel, iterations=1)

        return dilated

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        yellow_threshold = self.find_yellow_parts(self.cv_hsv_image)
        big_yellow_parts = self.find_big_parts(yellow_threshold)

        x, y, w, h = cv2.boundingRect(big_yellow_parts)

        left_top = (x, y)
        right_bottom = (x + w, y + h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.cv_bgr_image is not None:
                print "here"
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
