import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.linalg import norm
from scipy import sum, average
import ipdb
import os

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm

The TemplateMatcher class stores the template images along with keypoints
The predict method takes in a new image and returns a dictionary of the
different template images and how well the input image matched each of them

The _compute_prediction method is used by the predict method to find how
well the input image matches a given template.
_compute_prediction uses compare_images and normalize to quantify how
different two images are.
"""

def compare_images(img1, img2):
    """
    Find the difference between two images and quantify the difference in a
    single number
    """
    # normalize to compensate for exposure difference, this may be unnecessary
    # TODO: Find the difference beween the two images

def normalize(arr):
    """
    normalize the image array by taking (each element - mean) / standard dev
    This adjusts for differences in lighting
    """
    # TODO: normalize the input image


class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold

        # we have not tweaked, taken directly from tutorial
        self.flann_index_kdtree = 0
        self.trees = 5
        self.checks = 50
        self.ransac_thresh = 5.0
        self.index_params = dict(algorithm = self.flann_index_kdtree, trees = self.trees)
        self.search_params = dict(checks = self.checks)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)

            # precompute keypoints and descriptors for the template sign 
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        
        Iteratively call _compute_prediction to put together comparisons of one image with each template
        """
        #maps from the image keys of the template images to the visual
        #difference between the input image and that template
        visual_diff = {}

        try:
            # get keypoints and descriptors from input image using SIFT
            # TODO: get keypoints

            for k in self.signs.keys():
                #cycle trough templage images (k) and get the image differences
                visual_diff[k] = self._compute_prediction(k, img, kp, des)
        except:
            # could not find a homography, because the cropped image is bad.
            visual_diff = None

        if visual_diff:
            pass
            # TODO: convert visual_diff from values representing the difference
            #       between two images to template_confidence: confidence values 
            #       for how likely the image is to be template image k

        # if visual diff was not computed (bad crop, homography could not be computed)
        else:
            # set 0 confidence for all signs
            pass
            # TODO: delete the pass and indent the following line once the 
            #       if statement is written
        template_confidence = {k: 0 for k in self.signs.keys()}

        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison
        img: scene image
        kp: keypoints from scene image
        des: descriptors from scene image
        """

        # TODO: find corresponding points in the input image and the templae image

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
        
        visual_diff = compare_images(img_T, self.signs[k])

        
        # visual difference visualization and debugging
        # uncomment the following lines in order to visualize the difference computations

        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        # norm_im_T = normalize(img_T)
        # norm_sign = normalize(self.signs[k])
        # ax1.imshow(norm_im_T, cmap='gray')
        # ax1.set_title(img_T.dtype)
        # ax2.imshow(norm_sign, cmap='gray')
        # ax2.set_title(self.signs[k].dtype)
        # ax3.imshow(normalize(img_T) - normalize(self.signs[k]), cmap='gray')
        # ax3.imshow(norm_im_T - norm_sign, cmap='gray')
        # ax3.set_title("visual diff: %d" % visual_diff)
        # plt.title("should be" + k)
        # plt.xlabel(visual_diff)
        # plt.ylabel(visual_diff)
        # plt.show()

        return visual_diff


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

    for filename in scenes[:2]:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)
        print filename
        print pred