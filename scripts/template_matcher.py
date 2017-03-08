#!/usr/bin/env python

import cv2
import numpy as np
import os

import time

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""


class TemplateMatcher(object):

    def __init__(self, images, min_match_count=10, good_thresh=0.7, debug=False):
        self.signs = {}  # maps keys to the template images
        self.kps = {}  # maps keys to the keypoints of the template images
        self.descs = {}  # maps keys to the descriptors of the template images
        if cv2.__version__ == '3.1.0-dev':
            # this is totally going to break as soon as a new ROS version drops
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT()  # initialize SIFT to be used for image matching

        self.debug = debug

        if self.debug:
            cv2.namedWindow('img1')
            cv2.namedWindow('img2')
            cv2.moveWindow('img2', 600, 0)

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh  # use for keypoint threshold
        self.ransac_thresh = 5.0

        self.file_path = os.path.dirname(os.path.realpath(__file__))

        # TODO: precompute keypoints for template images
        for sign_name, filename in images.items():
            # load templates as grayscale

            self.signs[sign_name] = cv2.imread(os.path.join(self.file_path, filename), 0)

            # precompute keypoints and descriptors for the template sign
            self.kps[sign_name], self.descs[sign_name] = (
                self.sift.detectAndCompute(self.signs[sign_name], None)
            )

    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each
        template returns a dictionary, keys being signs, values being
        confidences
        """
        visual_diff = {}

        # TODO: get keypoints and descriptors from input image using SIFT
        #       store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img, None)

        for k in self.signs.keys():
            # cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # convert difference between images (from visual_diff)
            # to confidence values (stored in template_confidence)
            total = sum([1.0 / visual_diff[k] for k in self.signs.keys()])

            template_confidence = {k: 1.0 / visual_diff[k] / total for k in self.signs.keys()}
        else:
            # if visual diff was not computed
            # (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}

        return template_confidence

    def _compute_prediction(self, k, scene_img, scene_kps, scene_desc):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        scene_kps: keypoints from scene image,   scene_desc: descriptors from scene image
        """

        # TODO: find corresponding points in the input image and the template image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        bf = cv2.BFMatcher()
        # find SIFT descriptor matches in the images
        matches = bf.knnMatch(self.descs[k], scene_desc, k=2)

        # Apply Lowe Ratio Test to the keypoints
        # this should weed out unsure matches
        good_keypoints = []
        for m, n in matches:
            if m.distance < self.good_thresh * n.distance:
                good_keypoints.append(m)

        # transform the keypoint data into arrays for homography check
        # grab precomputed points
        template_pts = np.float32(
            [self.kps[k][m.queryIdx].pt for m in good_keypoints]
        ).reshape(-1, 1, 2)

        scene_img_pts = np.float32(
            [scene_kps[m.trainIdx].pt for m in good_keypoints]
        ).reshape(-1, 1, 2)

        # find out how to transform scene image to best match template
        M, mask = cv2.findHomography(scene_img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        # Transform input image so that it matches the template image as well as possible
        scene_img_T = cv2.warpPerspective(scene_img, M, self.signs[k].shape[::-1])

        visual_diff = self.compare_images(scene_img_T, self.signs[k])
        return visual_diff

    def _run_test(self):
        scenes = [
            "../images/uturn_scene.jpg",
            "../images/leftturn_scene.jpg",
            "../images/rightturn_scene.jpg"
        ]

        for filename in scenes:
            scene_img = cv2.imread(os.path.join(self.file_path, filename), 0)
            pred = tm.predict(scene_img)
            print filename.split('/')[-1]
            print pred

    def compare_images(self, img1, img2):
        if self.debug:
            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            cv2.waitKey(5)
            time.sleep(2)

        # find the mean squared difference between the images
        # http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        err = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        err /= float(img1.shape[0] * img2.shape[1])

        # lower is more similar
        return err

if __name__ == '__main__':
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
    }

    tm = TemplateMatcher(images, debug=False)

    tm._run_test()
