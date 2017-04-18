import rospy
import sensor_msgs.msg
from sign_localizer.msg import SignLocation

import cv2
from cv_bridge import CvBridge

import abc

class sign_follower(object):
    def __init__(self, node_name):
        rospy.init_node(node_name)
	self.debug_img = None
	self.img_bridge = CvBridge()
        self.img_sub = rospy.Subscriber('/camera/image_raw',
                                        sensor_msgs.msg.Image,
                                        self.on_new_image)
        self.sign_loc_pub = rospy.Publisher('/predicted_sign',
                                            SignLocation,
					    queue_size=10)

    @abc.abstractmethod
    def train(self, imdir):
	"""builds a set of features for each image in imdir"""
	pass

    @abc.abstractmethod
    def on_new_image(self, msg):
	"""compares the new image to the existing set of features
	   publishing the normalized probabilities of each image."""
	pass

    def run(self, rate):
	r = rospy.Rate(rate)
	while not rospy.is_shutdown():
	    if not self.debug_img is None:
		cv2.imshow("sign_debug", self.debug_img)
		cv2.waitKey(5)
	    r.sleep()


