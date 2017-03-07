#!/usr/bin/env python

import cv2
import os
import numpy as np

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold

        # precompute keypoints and descriptors for the template sign
        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)
            # compute keypoints and descriptors for the template sign 
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)

        #
        self.matcher = cv2.BFMatcher()


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        # compute keypoints and descriptors for the scene
        kp, des = self.sift.detectAndCompute(img,None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            template_confidence = {}
            for k in visual_diff:
            	template_confidence[k] = 50/visual_diff[k]

        else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}

        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        """
        ###taken from example code:

        #find matches
        matches = self.matcher.knnMatch(des,self.descs[k],k=2)

        # make sure the distance to the closest match is sufficiently better than the second closest
        good_matches = []
        for m,n in matches:      
            if (m.distance < 0.7*n.distance and
                kp[m.queryIdx].response > 0.0 and
                self.kps[k][m.trainIdx].response > 0.0):
                good_matches.append((m.queryIdx, m.trainIdx))

        #create pt lists
        img_pts = np.zeros((len(good_matches),2))
        tem_pts = np.zeros((len(good_matches),2))
        for idx in range(len(good_matches)):
            match = good_matches[idx]
            img_pts[idx,:] = kp[match[0]].pt
            tem_pts[idx,:] = self.kps[k][match[1]].pt

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, tem_pts, cv2.RANSAC, 8.0)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    """ determines how dissimilar skewed scene is to template"""
    #normalize scene pixel values
    img1_mean = img1.mean() 
    img1_std = img1.std()
    for i in np.nditer(img1,  op_flags=['readwrite']):
        i[...] = (i-img1_mean)/img1_std

    #normalize template pixel values
    img2_mean = img2.mean() 
    img2_std = img2.std()
    for i in np.nditer(img2, op_flags=['readwrite']):
        i[...] = (i-img2_mean)/img2_std

    #sums error
    error_array = img1 - img2
    error_array = error_array.astype(np.int8)
    ss_error = 0
    for i in np.nditer(error_array):
        ss_error += abs(i/255.0)**0.5
    #print ss_error
    return ss_error
   

if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }
    for key in images:
        path = os.path.join(dir_path, images[key])
        images[key] = path

    tm = TemplateMatcher(images)

    scenes = [
        "../images/uturn_box.png",
        "../images/leftturn_box.png",
        "../images/rightturn_box_small.png"
    ]
    for ind in range(0,len(scenes)):
        path = os.path.join(dir_path, scenes[ind])
        scenes[ind] = path
    
    for filename in scenes:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)
        print filename.split('/')[-1]
        print pred
