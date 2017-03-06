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
        cv2.namedWindow('post_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        #cv2.setMouseCallback('video_window', self.process_mouse_event)
        cv2.namedWindow('threshold_image')
        self.value_lower_bound = 152
        self.hue_lower_bound = 25
        self.sat_lower_bound = 157
        self.value_upper_bound = 255
        self.hue_upper_bound = 61
        self.sat_upper_bound = 255
        self.dilate_erode_iterations = 1
        self.moments = []


        cv2.createTrackbar('val lower bound', 'threshold_image', self.value_lower_bound, 255, self.set_value_lower_bound)
        cv2.createTrackbar('hue lower bound', 'threshold_image', self.hue_lower_bound, 255, self.set_hue_lower_bound)
        cv2.createTrackbar('sat lower bound', 'threshold_image', self.sat_lower_bound, 255, self.set_sat_lower_bound)
        cv2.createTrackbar('val upper bound', 'threshold_image', self.value_upper_bound, 255, self.set_value_upper_bound)
        cv2.createTrackbar('hue upper bound', 'threshold_image', self.hue_upper_bound, 255, self.set_hue_upper_bound)
        cv2.createTrackbar('sat upper bound', 'threshold_image', self.sat_upper_bound, 255, self.set_sat_upper_bound)
        cv2.createTrackbar('dilate erode iterations', 'threshold_image', self.dilate_erode_iterations, 5, self.set_dilate_erode_iterations)

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

    def set_value_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.value_lower_bound = val
    def set_hue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.hue_lower_bound = val
    def set_sat_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.sat_lower_bound = val
    def set_value_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.value_upper_bound = val
    def set_hue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.hue_upper_bound = val
    def set_sat_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.sat_upper_bound = val
    def set_dilate_erode_iterations(self, val):
        """ A callback function to handle the OpenCV slider to select the red lower bound """
        self.dilate_erode_iterations = val


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        kernel = np.ones((5,5),np.uint8)

        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.hsv_image, (self.hue_lower_bound,self.sat_lower_bound,self.value_lower_bound), (self.hue_upper_bound,self.sat_upper_bound,self.value_upper_bound))
        self.binary_image = cv2.erode(self.binary_image, kernel, iterations=self.dilate_erode_iterations)
        self.binary_image = cv2.dilate(self.binary_image, kernel, iterations=self.dilate_erode_iterations)
        moments = cv2.moments(self.binary_image)
        self.moments = moments
        #self.binary_image = cv2.inRange(self.cv_image, (128,128,128), (255,255,255))

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
        section_height = 30
        section_width = 40
        x_range = []
        y_range = []
        for height in range(0,480,section_height):
            for width in range(0,640,section_width):
                img_slice = self.binary_image[height:height+section_height,width:width+section_width]
                white_count = cv2.countNonZero(img_slice)
                if white_count > 50:
                    y_range.append(height)
                    x_range.append(width)
        y_range.sort()
        x_range.sort()
        left_top = (x_range[0], y_range[0])
        right_bottom = (x_range[-1]+section_width, y_range[-1]+section_height)
        #print x_range
        #print y_range
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                #print "here"
                # creates a window and displays the image for X milliseconds
                cv2.imshow('video_window', self.cv_image)
                #cv2.imshow('video_window', self.hsv_image)
                cv2.imshow('post_window', self.binary_image)
                #print self.moments
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':

    node = StreetSignRecognizer()
    node.run()
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }

    tm = TemplateMatcher(images)
    scenes = [
    "../images/uturn_scene.jpg",
    "../images/leftturn_scene.jpg",
    "../images/rightturn_scene.jpg"
]

for filename in scenes:
    scene_img = cv2.imread(filename, 0)
    pred = tm.predict(scene_img)
    print filename.split('/')[-1]
    print pred
