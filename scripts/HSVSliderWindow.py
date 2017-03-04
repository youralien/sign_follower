#!/usr/bin/env python
import cv2

class HSVSliderWindow(object):
    """Creates a slider that allows selection of HSV bounds"""

    
    def __init__(self):
        """ Initialize the HSV Slider Window"""
        self.window_name = 'threshold_image'

        cv2.namedWindow(self.window_name)
        self.hue_lower_bound = 0
        self.hue_upper_bound = 255
        self.sat_lower_bound = 0
        self.sat_upper_bound = 255
        self.val_lower_bound = 0
        self.val_upper_bound = 255


        cv2.createTrackbar('hue lower bound', self.window_name, 0, 255, self.set_hue_lower_bound)
        cv2.createTrackbar('hue upper bound', self.window_name, 0, 255, self.set_hue_upper_bound)
        cv2.createTrackbar('saturation lower bound', self.window_name, 0, 255, self.set_sat_lower_bound)
        cv2.createTrackbar('saturation upper bound', self.window_name, 0, 255, self.set_sat_upper_bound)
        cv2.createTrackbar('value lower bound', self.window_name, 0, 255, self.set_val_lower_bound)
        cv2.createTrackbar('value upper bound', self.window_name, 0, 255, self.set_val_upper_bound)

    def set_hue_lower_bound(self, val):
        self.hue_lower_bound = val
    def set_hue_upper_bound(self, val):
        self.hue_upper_bound = val
    def set_sat_lower_bound(self, val):
        self.sat_lower_bound = val
    def set_sat_upper_bound(self, val):
        self.sat_upper_bound = val
    def set_val_lower_bound(self, val):
        self.val_lower_bound = val
    def set_val_upper_bound(self, val):
        self.val_upper_bound = val