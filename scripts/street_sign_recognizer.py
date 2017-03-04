#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from HSVSliderWindow import HSVSliderWindow
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

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        left_top, right_bottom = self.sign_bounding_box(self.cv_image)
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def filter_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def sign_bounding_box(self, image):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        left_top = (200, 200)
        right_bottom = (400, 400)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""

        slider_window = HSVSliderWindow()
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print "here"
                filtered_image = self.filter_image(self.cv_image)
                cv2.imshow('video_window', filtered_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
