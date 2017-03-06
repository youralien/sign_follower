#!/usr/bin/env python

""" Detects and recognizes signs and publishes a ROS string message type describing the sign
(left,right,uturn)"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import template_matcher 
import operator

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign recognizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        cv2.namedWindow('hsv_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        cv2.namedWindow('threshold_image')
        self.hsv_lb = np.array([00, 190, 195]) # hsv lower bound for detection
        # #create trackbars for thresholding
        # #note while thresholding that worse detection (larger bounding boxes)
        # #creates better recognition results (no missing key points)
        # cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
        # cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
        # cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
        self.hsv_ub = np.array([30, 255, 255]) # hsv upper bound for detection
        # #create trackbars for theresholding
        # cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
        # cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
        # cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)

        #initialize template matcher
        self.images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }

    	self.tm = template_matcher.TemplateMatcher(self.images)

    	#tunable constant to only publish when you're sure
    	self.accepted_thresh = .4


        self.pub = rospy.Publisher('sign_detected', String, queue_size=10)


    #set bounds based on trackbars

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

        #read img, make hsv image, make thresholded image
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV) #Convert to HSV
        self.threshold_image = cv2.inRange(self.hsv_image,self.hsv_lb,self.hsv_ub) 


        #draw bounding box
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        #fudge factor to make the bounding box larger and capture keypoints 
        #at the very edge of the sign
        left = left - 10
        if left < 0:
        	left = 0
        top = top - 10
        if top < 0:
        	top = 0
        right = right + 10
        if right > 480:
        	right = 480
        bottom = bottom + 10
        if bottom > 640:
        	bottom = 640

        # crop bounding box region of interest
        self.cropped_sign = self.cv_image[top:bottom, left:right]

        #grayscale cropped sign for image comparison
        self.cropped_sign_gray = cv2.cvtColor(self.cropped_sign, cv2.COLOR_BGR2GRAY)


        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

        #print prediction
        pred = self.tm.predict(self.cropped_sign_gray)


        #compare images
        #if confidence is higher than accepted, print result
        if max([i for i in pred.values()]) > self.accepted_thresh:

        	recognized =  max(pred.iteritems(), key=operator.itemgetter(1))[0]
        	print recognized

        	self.pub.publish(recognized)



    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        #draw bounding rectangle

        x,y,w,h = cv2.boundingRect(self.threshold_image)

        left_top = (x,y)
        right_bottom = (x+w, y+h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                #print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('hsv_window', self.cropped_sign_gray)
                cv2.imshow('threshold_image', self.threshold_image)

                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()

