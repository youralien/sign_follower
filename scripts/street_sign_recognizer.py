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


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        images = {
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'
            }
        self.tm = TemplateMatcher(images)
        self.cv_image = None
        self.hsv_image = None
        self.binary_image = None                  # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.grid_size = (16, 16)
        cv2.namedWindow('original')
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        cv2.namedWindow('threshold_image')
        self.hsv_lb = np.array([0, 0, 0]) # hsv lower bound
        cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
        cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
        cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
        self.hsv_ub = np.array([255, 255, 255]) # hsv upper bound
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
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.box_height, self.box_width, self.channels = self.cv_image.shape
        self.box_width = self.box_width/self.grid_size[0]
        self.box_height = self.box_height/self.grid_size[1]
        self.filter()
        self.binary_image = cv2.inRange(self.hsv_image, (0, 105, 154), (255, 255, 255))
        self.detect_regions()
        #print(self.binary_image.item(84, 349))

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.binary_image, left_top, right_bottom, color=(255, 255, 255), thickness=5)

        self.tm.predict(cropped_sign)


    def filter(self):
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

    def detect_regions(self):
        self.region_list = []
        for i in range(self.grid_size[1]):
            tempList = []
            for j in range(self.grid_size[0]):
                if self.of_interest(j*self.box_width, i*self.box_height):
                    tempList.append(1)
                else:
                    tempList.append(0)
            self.region_list.append(tempList)


    def of_interest(self, start_x, start_y):
        count = 0
        #print(start_x, start_y)
        for i in range(start_y, start_y + self.box_height):
            for j in range(start_x, start_x + self.box_height):
                #print self.binary_image[i, j]
                if self.binary_image[i, j] > 0:
                    count += 1
        #print(count)
        if float(count)/(self.box_width*self.box_height) > 0.15:
            return True
        else:
            return False

    def get_bb(self):
        left = 1000
        top = 1000
        right = 0
        bottom = 0
        for i in range(self.grid_size[1]):
            for j in range(self.grid_size[0]):
                if self.region_list[i][j] == 1:
                    if i*self.box_height < top:
                        top = i * self.box_height
                    if j*self.box_width < left:
                        left = j*self.box_width
                    if i*self.box_height > bottom:
                        bottom = (i+1)*self.box_height
                    if j*self.box_width > right:
                        right = (j+1)*self.box_width
        return left, top, right, bottom


    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        left, top, right, bottom = self.get_bb()
        left_top = (left, top)
        right_bottom = (right, bottom)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image == None:
                # creates a window and displays the image for X milliseconds
                cv2.imshow('original', self.cv_image)
                cv2.imshow('video_window', self.hsv_image)
                cv2.imshow('threshold_image', self.binary_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
