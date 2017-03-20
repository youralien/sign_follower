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

    def __init__(self, templates):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.template_matcher = TemplateMatcher(templates)
        self.template_estimates = {}
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """

        #Connecting to the image publisher, then converting to HSV format
        #Then using HSV to identify the yellow in the sign (representing in binary_image)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.cv_image,(0,126,100),(50,255,255))

        #Find the edges of the sign, for cropping
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

        # predict which sign it is
        self.template_estimates = self.template_matcher.predict(cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY))

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        #Create list of points where the sign is
        points = np.transpose(np.nonzero(self.binary_image))
        #Get descriptors of boundry box
        x,y,w,h = cv2.boundingRect(points)

        #The processing underestimates sign area
        #so we need to add some padding with marg(margins)
        marg = 10
        y = max(y - marg, 0)
        x = max(x - marg, 0)
        w = min(w + 3*marg, self.binary_image.shape[1])
        h = min(h + 3*marg, self.binary_image.shape[0])

        #Identify points based on what was found
        #Have to transform coordinate system, x and y
        #are different for this bounding box.
        left_top = (y, x)
        right_bottom = (y+h, x+w)

        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print self.template_estimates
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('hsv_window', self.hsv_image)
                cv2.imshow('binary_window', self.binary_image)
                cv2.waitKey(5)
            r.sleep()
#End of StreetSignRecognizer Class

# I created this class to break out the hue and saturation adjustments
# They were unncecessary when running the main program, and made
# It unnecessarily ugly and full of methods.
class HSVSelector(object):
    """ Designed to help find ideal HSV values
        opens a binarized window with sliders to adjust the spectrum"""

    def __init__(self):
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

    #Collection of callback functions for sliders
    def set_hue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue lower bound """
        self.hue_lower_bound = val

    def set_hue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue upper bound """
        self.hue_upper_bound = val

    def set_saturation_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation lower bound """
        self.saturation_lower_bound = val

    def set_saturation_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation bound """
        self.saturation_upper_bound = val

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """

        #Connecting to the image publisher, then converting to HSV format
        #Then using HSV to identify the yellow (representing in binary_image)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.cv_image,
                (self.hue_lower_bound,self.saturation_lower_bound,100),
                (self.hue_upper_bound,self.saturation_upper_bound,255))

    def run(self):
        """ The main run loop"""
        rospy.init_node('HSV selector')
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
#End of HSVSelector Class

if __name__ == '__main__':
    #Images needed for finding the sign
    images = {
        "uturn": '../images/uturn_box_small.png',
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png'
        }
    node = StreetSignRecognizer(images)
    # node = HSVSelector()
    node.run()
