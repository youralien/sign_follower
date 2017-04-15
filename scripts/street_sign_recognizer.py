#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String
from template_matcher import TemplateMatcher, compare_images

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.hsv_image = None
        self.res_image = None
        self.mask = None
        self.cropped_sign = None
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        self.signpub = rospy.Publisher('/predicted_sign', String, queue_size=10)
        self.prediction = ""

        #sets up slider bars for hue isolation
        #cv2.namedWindow('threshold_image')
        # current values work well for uturn and right, but poorly for left
        # values for left (20, 166, 139) increase error for uturn and right
        self.hsv_lb = np.array([25, 202, 186]) # hsv lower bound 20, 166, 139 25, 202, 186 26, 214, 167 [15, 225, 139
        # cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
        # cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
        # cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
        self.hsv_ub = np.array([204, 255, 255]) # hsv upper bound 204, 255, 255 204, 255, 230
        # cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
        # cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
        # cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)

         # initialize template_matcher
        images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }
        
        self.tm = TemplateMatcher(images)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        #convert to img hsv
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        #create img mask for only yellow in hsv img
        self.mask = cv2.inRange(self.hsv_image, self.hsv_lb, self.hsv_ub)

        # Bitwise-AND mask and original image
        self.res_image = cv2.bitwise_and(self.cv_image,self.cv_image, mask= self.mask)

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        self.cropped_sign = cv2.cvtColor(self.cv_image[top:bottom, left:right], cv2.COLOR_BGR2GRAY) 


        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

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
        #use self.mask and boundingRect() to compute left_top and right_bottom
        bound = cv2.findNonZero(self.mask)
        (bx, by, bw, bh) = cv2.boundingRect(bound)
        left_top = (bx, by)
        right_bottom = (bx+bw, by+bh)
        return left_top, right_bottom

    def run(self):
        """ The main run loop sometimes throws errors, but if you try again it works"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print "here"
                if not self.cropped_sign is None:
                    pred = self.tm.predict(self.cropped_sign)
                    predNum = pred[max(pred, key=pred.get)]
                    #print predNum
                    if (predNum > 0.75):
                        self.prediction = max(pred, key=pred.get)
                        print self.prediction
                        print pred[self.prediction]
                    if not self.prediction == "":
                        self.signpub.publish(String(self.prediction))
                # creates a window and displays the image for X milliseconds
                #cv2.imshow('video_window', self.cv_image)
                #cv2.imshow('masked_window', self.res_image)
                #cv2.imshow('hsv_window', self.hsv_image)
                cv2.waitKey(5)
                #print "there"
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
