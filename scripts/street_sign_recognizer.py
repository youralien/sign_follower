#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy, os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import template_matcher as tm

class StreetSignRecognizer(object):
    """ This robot should recognize street signs """


    def __init__(self):
        """ Initialize the street sign reocgnizer """

        # Initialize template matcher 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        images = {
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'}
        for key in images:
            path = os.path.join(dir_path, images[key])
            images[key] = path
        self.matcher = tm.TemplateMatcher(images)
        

        # Initialize ROS message to OpenCV converter
        self.bridge = CvBridge()                    

        # Images and windows 
        self.cv_image = None
        self.hsv_image = None  		# cv_image transformed to hsv colorspace
        self.binary_image = None  	# masked image		
        cv2.namedWindow('video_window')
        #cv2.namedWindow('hsv_window')
        #cv2.namedWindow('threshold_image')
        #cv2.namedWindow('grid_window')

        #create mask bounds and gui
        self.hsv_lb = np.array([20, 180, 140]) # hsv lower bound
        cv2.createTrackbar('H lb', 'threshold_image', self.hsv_lb[0], 255, self.set_h_lb)
        cv2.createTrackbar('S lb', 'threshold_image', self.hsv_lb[1], 255, self.set_s_lb)
        cv2.createTrackbar('V lb', 'threshold_image', self.hsv_lb[2], 255, self.set_v_lb)
        self.hsv_ub = np.array([30, 255, 255]) # hsv upper bound
        cv2.createTrackbar('H ub', 'threshold_image', self.hsv_ub[0], 255, self.set_h_ub)
        cv2.createTrackbar('S ub', 'threshold_image', self.hsv_ub[1], 255, self.set_s_ub)
        cv2.createTrackbar('V ub', 'threshold_image', self.hsv_ub[2], 255, self.set_v_ub)

        # create ros node and pubscribers
        rospy.init_node('street_sign_recognizer')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # change to hsv colorspace
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)  

	# mask hsv space
        self.binary_image = cv2.inRange(self.hsv_image, self.hsv_lb, self.hsv_ub)    

        # determine bouning box
        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)

        #attempt to id sign
        try:
            self.check_sign(cropped_sign)
        except:
	    print("bad box")

    def check_sign(self,img):
        """calls template matcher for sign identification"""
        
        # change to greyscale colorspace
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        #calculate likelihoods
        pred = self.matcher.predict(img)
        print pred

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

        #grid parameters
        #width, height =self.cv_image.shape[:2]
        grid_size = (24,18)
	cell_w = 640/grid_size[0]
	cell_h = 480/grid_size[1]
        on_threshold = 0.25
        #grid = np.zeros(grid_size)

        #on counter
        rows = []
        columns = []

	# slice image and determine high white locations
        for column in range(0,grid_size[0]):
            for row in range(0,grid_size[1]):                
                cell = self.binary_image[row*cell_h:(row+1)*cell_h,
                                         column*cell_w:(column+1)*cell_w]
                white_ct = cv2.countNonZero(cell) 
                percent = 1.0*white_ct/(cell_w*cell_h)
                if percent > on_threshold:                    
		    #grid[row][column] = 1 #update grid for validation printing
                    rows.append(row)
                    columns.append(column)                          
	#print grid
        #print("------")
 
        #determine bounding box grid location
        left = min(columns)
        right = max(columns) +1
        top = min(rows)
        bottom = max(rows) +1 

        #determine bounding box pixel location
        left_top = (left*cell_w, top*cell_h)
        right_bottom = (right*cell_w, bottom*cell_h)
        return left_top, right_bottom

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            # creates windows and displays the image for 5 milliseconds
            if not self.cv_image is None:
                cv2.imshow('video_window', self.cv_image)
                #cv2.imshow('hsv_window', self.hsv_image)
                #cv2.imshow('threshold_image', self.binary_image)
		#cv2.imshow("grid_window", self.grid)
                cv2.waitKey(5)
            r.sleep()

if __name__ == '__main__':
    node = StreetSignRecognizer()
    node.run()


