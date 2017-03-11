#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from template_matcher import TemplateMatcher
from grid_image import GridImage

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        # self.grid_cell = None
        self.binary_image = None
        self.image_info_window = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        # self.hsv_lb = np.array([23, 175, 133])           # hsv lower bound
        # self.hsv_ub = np.array([40, 255, 255])     # hsv upper bound
        self.hsv_lb = np.array([0, 0, 0])           # hsv lower bound
        self.hsv_ub = np.array([255, 255, 255])     # hsv upper bound
        # self.grid_cell_w = 64*3
        # self.grid_cell_h = 48*3
        self.TM = TemplateMatcher() 
        self.grid = GridImage()

        cv2.namedWindow('video_window')
        cv2.setMouseCallback('video_window', self.process_mouse_event)
        cv2.namedWindow('threshold_image')
        
        cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
        cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
        cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
        
        cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
        cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
        cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)
        
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.good_thresh = cv2.threshold(self.cv_image, self.hsv_lb[0], self.hsv_ub[0], cv2.THRESH_BINARY)
        # # print type(self.good_thresh[1])
        # if not self.good_thresh is None:
        #     cv2.imshow('threshold_image',self.good_thresh[1])
        #     cv2.waitKey(5)
        
        # # NumPy array slicing!!
        # self.grid_cell = self.gray_image[self.grid_cell_h:2*self.grid_cell_h,
        #                 self.grid_cell_w:2*self.grid_cell_w]

        self.binary_image = cv2.inRange(self.cv_image, self.hsv_lb, self.hsv_ub)
        if not self.binary_image is None:
            cv2.imshow('threshold_image',self.binary_image)
            cv2.waitKey(5)

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.gray_image[top:bottom, left:right]
        # cropped_sign = cv2.rectangle(self.gray_image,left_top,right_bottom,(0,255,0),2)
        print self.TM.predict(self.gray_image)

        # draw bounding box rectangle
        cv2.rectangle(self.gray_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values associated
            with a particular pixel in the camera images """
        image_info_window = 255*np.ones((500,500,3))
        # show hsv values
        cv2.putText(self.image_info_window,
                    'Color (h=%d,s=%d,v=%d)' % (self.hsv_image[y,x,0], self.hsv_image[y,x,1], self.hsv_image[y,x,2]),
                    (5,50), # 5 = x, 50 = y
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))
        # show bgr values
        cv2.putText(image_info_window,
                    'Color (b=%d,g=%d,r=%d)' % (self.cv_image[y,x,0], self.cv_image[y,x,1], self.cv_image[y,x,2]),
                    (5,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))
        if not self.image_info_window is None:
            cv2.imshow('image_info', image_info_window)
            cv2.waitKey(5)

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

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # TODO: YOUR SOLUTION HERE
        self.binary_image = cv2.inRange(self.cv_image, self.hsv_lb, self.hsv_ub)
        _, self.good_thresh = cv2.threshold(self.cv_image, self.hsv_lb[2], self.hsv_ub[2], cv2.THRESH_BINARY)
        
        contours,_ = cv2.findContours(self.binary_image, 1, 2)
        # print type(contours[0])
        area = 0
        max_cnt = None
        for cnt in contours:
            if area < cv2.contourArea(cnt):
                area = cv2.contourArea(cnt)
                max_cnt = cnt
        # print max_cnt

        x,y,w,h = cv2.boundingRect(max_cnt)
        print w,h
        # x,y,w,h = cv2.boundingRect(np.array(self.binary_image, dtype=int))

        for i in self.grid.grid_cell

        left_top = (x,y)
        right_bottom = (x+w,y+h)

        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None; # and not self.grid_cell is None:
                print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.gray_image)
                # cv2.imshow('video_window', self.grid_cell)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()