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
        #self.slider_window = HSVSliderWindow()
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

    def filter_image(self, image, h_l=27, h_u=31, s_l=166, s_u=255, v_l=160, v_u=243):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        binary_image = cv2.inRange(hsv_image, (h_l,s_l,v_l), (h_u,s_u,v_u))
        return binary_image


    def sign_bounding_box(self, image):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # h_l = self.slider_window.hue_lower_bound
        # h_u = self.slider_window.hue_upper_bound
        # s_u = self.slider_window.sat_upper_bound
        # s_l = self.slider_window.sat_lower_bound
        # v_l = self.slider_window.val_lower_bound
        # v_u = self.slider_window.val_upper_bound
        binary_image = self.filter_image(image)
        contours, _ = cv2.findContours(binary_image, 0, 2)
        cv2.drawContours(self.cv_image, contours, -1, (255,0,0), 3)

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        left_top = (x, y)
        right_bottom = (x+w, y+h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
