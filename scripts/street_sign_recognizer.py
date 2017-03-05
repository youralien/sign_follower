#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from template_matcher import TemplateMatcher

class StreetSignRecognizer(object):
    """This robot should take take in an image of a street sign and match that
       with what street sign it is.
    """

    def __init__(self):
        """Initialize the street sign recognizer """
        rospy.init_node('street_sign_recognizer')
        rospy.Subscriber("sign_follower/cropped_sign", Image, self.recognize_sign)

        images = {
            "left": '../images/leftturn_box_small.png',
            "right": '../images/rightturn_box_small.png',
            "uturn": '../images/uturn_box_small.png'
            }
        self.tm = TemplateMatcher(images)

    def recognize_sign(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        pred = self.tm.predict(image)
        print pred

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(9)
        while not rospy.is_shutdown():
            print "asdf"
            r.sleep()

