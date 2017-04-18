from __future__ import division
from classes import sign_follower

import os

import numpy as np

import cv2
from cv2.xfeatures2d import SIFT_create

from sign_localizer.msg import SignLocation

class sift_flann_localizer(sign_follower):
    def __init__(self):
	super(self.__class__, self).__init__('dhash_sift_flann_localizer')
	self.sift = SIFT_create()
	self.flann = cv2.FlannBasedMatcher_create()
	self.images = self.train('../images/train/')
	self.feature_threshold = 200

    def train(self, imdir):
	assoc = {}
	for img_name in os.listdir(imdir):
	    if not img_name.endswith('.png'):
		continue
	    img = cv2.imread(imdir+img_name, cv2.IMREAD_GRAYSCALE)
	    assoc[img_name] = self.sift.detectAndCompute(img, None)
	return assoc
	     
    def on_new_image(self, msg):
	#convert the image
	img = self.img_bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
	#SIFT the image
	kp, desc = self.sift.detectAndCompute(img, None)
	#get the costs of each of the keypoint matches
	#there's a phase change as keypoint distances increase to invalid
	#if you plot this you'll see it.
	#punting on the problem of determining this phase change (2 parameter model)
	#instead thresholding and counting
	matches = {name : sorted([x[0] for x in self.flann.knnMatch(desc, d, k=1)], key=lambda x: x.distance)\
			  for name, (_, d) in self.images.iteritems()}
	counts = {name: sum([k.distance for k in kps[:self.feature_threshold]]) for name, kps in matches.iteritems()}
	tot_dist = sum(counts.values())
	rel_probs = {name: 1-(dist/tot_dist) for name, dist in counts.iteritems()}
	max_class = sorted(rel_probs.iteritems(), key=lambda kv: kv[1], reverse=True)[0][0]
	max_class_features = matches[max_class][:self.feature_threshold]
	centroid = [int(pt) for pt in np.mean(np.array([kp[f.trainIdx].pt for f in max_class_features]), axis=0)]
	cv2.circle(img, tuple(centroid), 10, (255, 0, 0))
	self.debug_img = img
	self.sign_loc_pub.publish(SignLocation(header=msg.header,
					       x=centroid[0],
					       y=centroid[1],
					       names=rel_probs.keys(),
					       rel_probs=rel_probs.values()))

	
