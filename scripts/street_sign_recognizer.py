#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import template_matcher as tm

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')             # used to display messages from the neato camera
        cv2.namedWindow('post_window')              # after the image is processed
        cv2.namedWindow('threshold_image')          # used to house sliders for tweaking image filtering
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        self.images = {                             # load images that the feed will be compared to
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'
            }

        #default values for filtering (what has worked before)
        self.value_lower_bound = 152
        self.hue_lower_bound = 25
        self.sat_lower_bound = 157
        self.value_upper_bound = 255
        self.hue_upper_bound = 61
        self.sat_upper_bound = 255
        self.dilate_erode_iterations = 1

        #setup trackbars for tweaking
        cv2.createTrackbar('val lower bound', 'threshold_image', self.value_lower_bound, 255, self.set_value_lower_bound)
        cv2.createTrackbar('hue lower bound', 'threshold_image', self.hue_lower_bound, 255, self.set_hue_lower_bound)
        cv2.createTrackbar('sat lower bound', 'threshold_image', self.sat_lower_bound, 255, self.set_sat_lower_bound)
        cv2.createTrackbar('val upper bound', 'threshold_image', self.value_upper_bound, 255, self.set_value_upper_bound)
        cv2.createTrackbar('hue upper bound', 'threshold_image', self.hue_upper_bound, 255, self.set_hue_upper_bound)
        cv2.createTrackbar('sat upper bound', 'threshold_image', self.sat_upper_bound, 255, self.set_sat_upper_bound)
        cv2.createTrackbar('dilate erode iterations', 'threshold_image', self.dilate_erode_iterations, 5, self.set_dilate_erode_iterations)

        #init matcher from other half of project
        self.matcher = tm.TemplateMatcher(self.images)
        self.pred = {}

    #callback functions for sliders

    def set_value_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the value lower bound """
        self.value_lower_bound = val
    def set_hue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue lower bound """
        self.hue_lower_bound = val
    def set_sat_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation lower bound """
        self.sat_lower_bound = val
    def set_value_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the value upper bound """
        self.value_upper_bound = val
    def set_hue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue upper bound """
        self.hue_upper_bound = val
    def set_sat_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the saturation upper bound """
        self.sat_upper_bound = val
    def set_dilate_erode_iterations(self, val):
        """ A callback function to handle the OpenCV slider to select the number of times to dilate and erode"""
        self.dilate_erode_iterations = val

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image. Converts to binary, erodes, and dilates for cropping. Calls cropping, then passes to matcher"""

        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        kernel = np.ones((5,5),np.uint8)

        #filter image
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV) #change from rgb to hsv
        self.binary_image = cv2.inRange(self.hsv_image, (self.hue_lower_bound,self.sat_lower_bound,self.value_lower_bound), (self.hue_upper_bound,self.sat_upper_bound,self.value_upper_bound))
        self.binary_image = cv2.erode(self.binary_image, kernel, iterations=self.dilate_erode_iterations)
        self.binary_image = cv2.dilate(self.binary_image, kernel, iterations=self.dilate_erode_iterations)

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = cv2.cvtColor(self.cv_image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

        self.pred = self.matcher.predict(cropped_sign) #predict using template matcher



    def sign_bounding_box(self):
        """
        Sections up image into grid, analyzes grid sections, uses grid to return (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
        defining topleft and bottomright corners of the bounding box
        """
        #how big sections of grid are
        section_height = 30
        section_width = 40
        #lists to contain section with stuff in them
        x_range = []
        y_range = []
        #slice into grid
        for height in range(0,480,section_height):
            for width in range(0,640,section_width):
                img_slice = self.binary_image[height:height+section_height,width:width+section_width]
                white_count = cv2.countNonZero(img_slice)
                if white_count > 50:    #threshold for how much white (yellow) in binary image
                    y_range.append(height)
                    x_range.append(width)
        y_range.sort()
        x_range.sort()
        left_top = (x_range[0], y_range[0])
        right_bottom = (x_range[-1]+section_width, y_range[-1]+section_height)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                # creates two window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('post_window', self.binary_image)
                cv2.waitKey(5)
                print max(self.pred) #prediction values for the sign
            r.sleep()

if __name__ == '__main__':

    node = StreetSignRecognizer()
    node.run()
