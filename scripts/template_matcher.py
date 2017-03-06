#!/usr/bin/env python

import cv2
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
        self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        self.min_match_count = min_match_count # Number of good matches needed to transform image
        self.good_thresh = good_thresh #use for keypoint threshold

        cv2.namedWindow('transformed image')

        #Precompute keypoints for template images
        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)
            # precompute keypoints and descriptors for the template sign 
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)

    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        # Get keypoints and descriptors from input image using SIFT
        #       store keypoints in variable kp and descriptors in des

        kp, des = self.sift.detectAndCompute(img,None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)
        
        template_confidence = {}
        if visual_diff:
            # Convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)
            for k in visual_diff.keys():
                if visual_diff[k]<=0:
                    confidence = 0
                else:
                    confidence = round(visual_diff[k],2)
                template_confidence[k] = confidence 
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
        # Find corresponding key points in the input image and the template image using cv2's Flann Based algorithm
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des,self.descs[k],k=2)

        # store all the good matches as per Lowe's ratio test. From experience, most correct matches satisfy this test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance: #Tuning this value changes the behavior a lot, 0.7 is definitely an optimum
                good.append(m)

        if len(good)>self.min_match_count:
            # put keypoints from template image in template_pts
            # put corresponding keypoints from input image in img_pts
            img_pts = np.float32([kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            template_pts = np.float32([self.kps[k][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.good_thresh)
            img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

            #Visualize the image to see if it looks like the template:
            cv2.imshow('transformed image', img_T)
            cv2.waitKey(0) #Wait for a key to be pressed to move on

            visual_diff = compare_images(img_T, self.signs[k])
            return visual_diff
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            return None

# end of TemplateMatcher class

def compare_images(img1, img2):
    """Find the correlation coefficient between img1 and img2
    Returns a value from -1 to 1"""
    img1 = np.ravel(normalize(img1))
    img2 = np.ravel(normalize(img2))
    return np.corrcoef(img1,img2)[0,1] #this matrix is 2 by 2 

def normalize(img):
    """Returns a normalized img matrix"""
    return(img-np.mean(img))/np.std(img)

if __name__ == '__main__':
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