#!/usr/bin/env python

"""
This code allows the neatos to recognize and obey yellow traffic signs.
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3

class SignFollower(object):
    def __init__(self):
        rospy.init_node('sign_follwer')