#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from color_slider import Color_Slider

class StreetSignRecognizer(Color_Slider, object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        #
        # initialize the color slider
        # super(StreetSignRecognizer, self).__init__()

        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

        # uncomment for viewing hsv values in image
        # self.image_info_window = None
        # cv2.setMouseCallback('video_window', self.process_mouse_event)


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.hsv_image, (20,135,167), (34,255,238))

        # uncomment for color sliders
        # self.binary_image = cv2.inRange(self.hsv_image, (self.hsv_lb[0],self.hsv_lb[1],self.hsv_lb[2]), (self.hsv_ub[0],self.hsv_ub[1],self.hsv_ub[2]))


        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # TODO: YOUR SOLUTION HERE

        # set level of connectivity
        connectivity = 4

        # comute connected regions
        output = cv2.connectedComponentsWithStats(self.binary_image, connectivity, cv2.CV_32S)
        # The third cell is the stat matrix
        stats = output[2]

        # get row index of second highest conneted region
        idx = np.argmax(stats[1:,4]) + 1

        # extract the bounding box
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        w = stats[idx, cv2.CC_STAT_WIDTH]

        # extract smallest square feature, also removes stand
        if h > w:
            h = w
        elif w > h:
            w = h

        # calculate bounding box position
        left_top = (x, y)
        right_bottom = (x + w, y + h)

        return left_top, right_bottom


    # def process_mouse_event(self, event, x,y,flags,param):
    #     """ Process mouse events so that you can see the color values associated
    #     with a particular pixel in the camera images """
    #     self.image_info_window = 255*np.ones((500,500,3))
    #
    #     # show hsv values
    #     cv2.putText(self.image_info_window,
    #     'Color (h=%d,s=%d,v=%d)' % (self.hsv_image[y,x,0], self.hsv_image[y,x,1], self.hsv_image[y,x,2]),
    #     (5,50), # 5 = x, 50 = y
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0,0,0))


    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.waitKey(5)
                cv2.imshow('binary_window', self.binary_image)
                cv2.waitKey(5)
            # if not self.image_info_window is None:
            #     cv2.imshow('image_info', self.image_info_window)
            #     cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
