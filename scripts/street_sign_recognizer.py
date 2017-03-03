#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
        cv2.setMouseCallback('video_window', self.process_mouse_event)
        cv2.namedWindow('hsv_video_window')
        cv2.setMouseCallback('hsv_video_window', self.process_mouse_event)
        cv2.namedWindow('binary_video_window')
        self.binary_image = 0
        self.hsv_image = 0
        # Initialize for bounding box
        self.box_x = 0
        self.box_y = 0
        self.box_w = 0
        self.box_h = 0

        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

  

    def set_twist(self):
        """ Setting the Twist velocity of the robot using proportional control. """
        self.kp = .005
        window_x = 320
        moments = cv2.moments(self.binary_image)
        if moments['m00'] != 0:
            self.center_x, self.center_y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
            diff_x = self.center_x - window_x
            print diff_x

            if math.fabs(diff_x) < 20:
                self.twist.linear.x = .3
                self.twist.angular.z = 0

            else:
                #self.twist.linear.x = math.fabs(1/diff_x)*5
                self.twist.linear.x = 0
                self.twist.angular.z = -diff_x*self.kp


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        
        self.binary_image = cv2.inRange(self.cv_image, (0, 180, 200), (30, 255, 255))
        #self.binary_image = cv2.inRange(self.hsv_image, (0, 180, 180), (100, 255, 255))

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values associated
            with a particular pixel in the camera images """
        image_info_window = 255*np.ones((500,500,3))
        cv2.putText(image_info_window,
                    'Color (b=%d,g=%d,r=%d)' % (self.cv_image[y,x,0], self.cv_image[y,x,1], self.cv_image[y,x,2]),
                    (5,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))
        cv2.imshow('image_info', image_info_window)
        cv2.waitKey(5)

    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        contours,hierarchy = cv2.findContours(self.binary_image, 1, 2)
        cnt = contours[0]
        moments = cv2.moments(self.binary_image)
        if moments['m00'] != 0:
            self.center_x, self.center_y = int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
            #self.box_x,self.box_y,self.box_w,self.box_h = cv2.boundingRect(cnt)
            
        #left_top = (200, 200)
        #right_bottom = (400, 400)

        #left_top = (self.box_x, self.box_y)
        #right_bottom = (self.box_x + self.box_w, self.box_y + self.box_h)
        left_top = (self.center_x - 100, self.center_y-100)
        right_bottom = (self.center_x + 100, self.center_y +100)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                cv2.imshow('binary_video_window', self.binary_image)
                cv2.imshow('hsv_video_window', self.hsv_image)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()
