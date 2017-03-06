#!/usr/bin/env python

import cv2
import numpy as np

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""


class TemplateMatcher(object):

    def __init__(self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {}  # maps keys to the template images
        self.kps = {}  # maps keys to the keypoints of the template images
        self.descs = {}  # maps keys to the descriptors of the template images
        if cv2.__version__ == '3.1.0-dev':
            # this is totally going to break as soon as a new ROS version drops
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT()  # initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh  # use for keypoint threshold
        self.ransac_thresh = 5.0

        # TODO: precompute keypoints for template images
        for sign_name, filename in images.items():
            # load templates as grayscale
            self.signs[sign_name] = cv2.imread(filename, 0)

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

        print visual_diff

        if visual_diff:
            pass
            # TODO: convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)

        else:  # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}

        # TODO: delete line below once the if statement is written
        template_confidence = {k: 0 for k in self.signs.keys()}

        return template_confidence

    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        """

        # TODO: find corresponding points in the input image and the template image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, self.descs[k], k=2)

        # Apply ratio test (from opencv-python-tutroals (sic))
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        img_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        template_pts = np.float32([self.kps[k][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # grab precomputed points

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        # TODO: change img to img_T once you do the homography transform
        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff

    def _run_test(self):
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


def compare_images(img1, img2):
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(5)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    cv2.namedWindow('img1')
    cv2.namedWindow('img2')
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
    }

    tm = TemplateMatcher(images)

    tm._run_test()
