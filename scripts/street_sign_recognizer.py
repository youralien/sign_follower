#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Bool
import cv2
import numpy as np
import os

# TODO: uncomment this line once you have TemplateMatcher working
# from template_matcher import TemplateMatcher


class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    curr_dir = os.path.dirname(os.path.realpath(__file__))
    template_images = {
        "lturn":  os.path.join(curr_dir, '../images/leftturn_box_small.png'),
        "rturn": os.path.join(curr_dir, '../images/rightturn_box_small.png'),
        "uturn": os.path.join(curr_dir, '../images/uturn_box_small.png')
    }

    def __init__(self, image_topic, sleep_topic):
        """ Initialize the street sign reocgnizer """
        rospy.init_node('street_sign_recognizer')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        self.saveCounter = 0                        # how many images we've saved to disk


        # TODO: uncomment these lines once you have TemplateMatcher working
        # print "Loading TemplateMatcher"
        # self.template_matcher = TemplateMatcher(self.template_images)

        self.pub = rospy.Publisher('predicted_sign', String, queue_size=1)
        cv2.namedWindow('video_window')

        self.sleeping = False

        self.running_predictions = {"lturn": 0, "rturn": 0, "uturn": 0}

        # 'use' parameters for quick changes in node functionality
        self.use_slider = False
        self.use_mouse_hover = False
        self.use_saver = False
        self.use_predict = False

        # threshold by which the running confidence summation must achieve to publish a predicted_sign
        # TODO: hand tune this value with testing
        self.decision_threshold = 35

        if self.use_mouse_hover:
            # when mouse hovers over video window
            cv2.setMouseCallback('video_window', self.process_mouse_event)

        if self.use_slider:
            cv2.namedWindow('threshold_image')
            self.hsv_lb = np.array([0, 0, 0])
            cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
            cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
            cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
            self.hsv_ub = np.array([255, 255, 255])
            cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
            cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
            cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)

        rospy.Subscriber(image_topic, Image, self.process_image)
        rospy.Subscriber(sleep_topic, Bool, self.process_sleep)

    # # # # # # # # # #
    # color callbacks #
    # # # # # # # # # #

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

    def process_sleep(self, msg):
        """ Process sleep messages from the navigation node and stash them
            in an attribute called sleeping """
        self.sleeping = msg.data

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        pt1, pt2 = self.sign_bounding_box()

        # crop and gray scale the bounding box region of interest
        cropped_sign = self.cv_image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        cropped_sign_gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, pt1, pt2, color=(0, 0, 255), thickness=5)
 
        # creates a window and displays the image for a X milliseconds 
        cv2.imshow('video_window', self.cv_image)
        cv2.waitKey(5)

        # saving frames of video as images can be a useful format to work with
        # i.e. if you want to load the image frame in a jupyter notebook
        if self.use_saver:
            cv2.imwrite("/tmp/bin_img_{0:0>4}.jpg".format(self.saveCounter), cropped_sign)
            self.saveCounter += 1

        if self.use_predict and pt1 and pt2:
            # make predictions with confidence for each sign key
            # TODO: uncomment the predict call once template matcher is working
            # prediction = self.template_matcher.predict(cropped_sign_gray)
            
            # TODO: remove this dummy prediction
            prediction = {'rturn': 0.0, 'lturn': 0.0, 'rturn': 0.0}

            for sign_key in prediction:
                self.running_predictions[sign_key] += prediction[sign_key]

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values associated
            with a particular pixel in the camera images """
        image_info_window = 255*np.ones((500,500,3))

        # show bgr values
        cv2.putText(image_info_window,
                    'Color (b=%d,g=%d,r=%d)' % (self.cv_image[y,x,0], self.cv_image[y,x,1], self.cv_image[y,x,2]),
                    (5,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))

        # TODO: show hsv values

        cv2.imshow('image_info', image_info_window)
        cv2.waitKey(5)

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if not self.sleeping:
                # TODO: publish to the /predicted_sign topic once confident
                # running_predictions is dictionary, with
                # keys being 'lturn', 'rturn', and 'uturn',
                # values being cumulative prediction confidences.
                # Use some logic like...
                # if running_predictions[sign_key] > self.decision_threshold:
                #   publish(sign_key)
                #   reset running_predictions to 0 again
                pass
            else:
                # Don't store running predictions while sleeping
                self.running_predictions = {"lturn": 0, "rturn": 0, "uturn": 0}

            r.sleep()

    def sign_bounding_box(self):
        """
        Returns
        -------
        (pt1, pt2) where pt1 and pt2 are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box 
        """
        # TODO: YOUR SOLUTION HERE
        pt1 = (200, 200)
        pt2 = (400, 400)
        return pt1, pt2
    
if __name__ == '__main__':
    node = StreetSignRecognizer("/camera/image_raw","/imSleeping")
    node.run()
