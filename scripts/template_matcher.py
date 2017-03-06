import cv2
import numpy as np
import math
import sys
import time

"""
CompRobo Spring 2017

Completed by Kevin Zhang

This code determines which of a set of template images matches
an input image the best using the SIFT algorithm. Steps involved:

1. Finding the keypoints of the input image
2. Matching the keypoints of the input image with the template_pts
3. Determining the best fit images using euclidean distance and ratio test
4. Generate and return a set of confidence values on each template image's match
to the input

"""

class TemplateMatcher(object):
    """
    The main class for matching templates to input images, which holds the sift
    algorithm, euclidean distance and confidence values
    """
    def __init__ (self, images, min_match_count=10, good_thresh=0.75):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images

        #initialize SIFT module
        self.sift = cv2.xfeatures2d.SIFT_create()

        #the final image used in comparison after matching keypoints and warping
        self.img_T = None

        #variables for determining good keypoints/image matching
        self.min_match_count = min_match_count #minimum number of keypoints needed for an image to be good
        self.good_thresh = good_thresh #use for keypoint threshold
        self.ransac_thresh = 5 #use for finding homography of image

        #creates database of keypoints and descriptors on template images
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
        #dictionary that holds the key to each image's matrix difference from the input
        visual_diff = {}

        #compute the keypoints and descriptors of the input image using SIFT
        kp, des = self.sift.detectAndCompute(img, None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            difference = self._compute_prediction(k, img, kp, des)
            #if the difference is none, then set that confidence value to 0 (using inf)
            if difference is None:
                difference = np.inf
            visual_diff[k] = difference

        if visual_diff:
            #generate confidence value dictionary by taking inverse of each visual diff
            template_confidence = {k:1/visual_diff[k] for k in visual_diff.keys()}
            #normalize the dictionary
            temp_sum = np.sum(template_confidence.values())
            if temp_sum != 0:
                for k in template_confidence.keys():
                    template_confidence[k] /= temp_sum
            else: #if the sum of the dictionary values is invalid(all sum to 0, nan, etc.)
                #set 0 confidence for all signs
                template_confidence = {k: 0 for k in self.signs.keys()}

        else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}

        #note that there is a bit of repetition in the above code, however I believe it must be this way
        #because there must be multiple checks to make sure that deeper calculations are still okay

        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        """
        #initialize the two lists of corresponding matching keypoints
        template_pts = []
        img_pts = []

        #iterate through all keypoints of the template image to find matching keypoints to them
        for i in range(len(self.kps[k])):
            distances = []  #list of euclidean distances between all keypoints of input and current template keypoint
            for j in range(len(kp)):
                #compute the euclidean distance between two 128 dimensional vectors, and add to distances along with keypoints
                distances.append((np.linalg.norm(self.descs[k][i] - des[j]), self.kps[k][i], kp[j]))

            #sort the list and extract the two smallest distances
            distances = sorted(distances)
            m, n = distances[0], distances[1]

            #check if a good match based on ratio test
            if m[0] < self.good_thresh * n[0]:
                template_pts.append(m[1].pt)
                img_pts.append(m[2].pt)

        #check that there is a sufficient amount of matching keypoints
        if len(img_pts) >= self.min_match_count:
            img_pts = np.asarray(img_pts, dtype=float)
            template_pts = np.asarray(template_pts, dtype=float)

            #use findHomography and warpPerspective to morph the input image using matching keypoints
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
            #the idea is that img_T should be very similar to one of the three templates
            self.img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

            #compute the direct difference between the morphed image and the template
            visual_diff = self.compare_images(self.img_T, self.signs[k])
            return visual_diff
        else:  #if there are insufficient matching keypoints
            #ignore this iteration by returning none
            return None


    def compare_images(self,img1, img2):
        """
        compares the two images based on their matrices, and returns the literal
        difference between the two
        """
        #normalizes the two images
        img1_mean = np.mean(img1)
        img1_std = np.std(img1)
        img2_mean = np.mean(img2)
        img2_std=  np.std(img2)
        norm_img1 = (img1 - img1_mean)/img1_std
        norm_img2 = (img2 - img2_mean)/img2_std

        #returns the differences of the norms
        return np.sum(abs(norm_img2 - norm_img1))



if __name__ == '__main__':
    #dictionary of template images
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }

    #initializes the TemplateMatcher class using the above dictionary
    tm = TemplateMatcher(images)

    #test images
    scenes = [
    "../images/uturn_scene.jpg",
    "../images/leftturn_scene.jpg",
    "../images/rightturn_scene.jpg"
    ]

    #for each test image, predict the sign
    for filename in scenes:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)

        print filename.split('/')[-1]
        print pred
        print ' '
