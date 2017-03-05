#!/usr/bin/env python
from time import sleep

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""


class TemplateMatcher(object):
    def __init__(self, images, min_match_count=10, good_thresh=200, ransac_thresh=1):
        self.signs = {}  # maps keys to the template images
        self.kps = {}  # maps keys to the keypoints of the template images
        self.descs = {}  # maps keys to the descriptors of the template images
        self.neighbors = {}  # maps keys to scikitlearn neighbor data structures
        if cv2.__version__ == '3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT()  # initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh  # use for keypoint threshold
        self.ransac_thresh = ransac_thresh

        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename, 0)

            # precompute keypoints and descriptors for the template sign
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k], None)

            self.neighbors[k] = NearestNeighbors(n_neighbors=1, algorithm="ball_tree"). \
                fit(self.descs[k])

        cv2.namedWindow('template')
        cv2.namedWindow('match')


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        kp, des = self.sift.detectAndCompute(img, None)
        #       store keypoints in variable kp and descriptors in des

        for k in self.signs.keys():
            # cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

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

        # print kp[:5], des[:5], np.shape(des)


        # Done: find corresponding points in the input image and the template image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts


        distances, indices = self.neighbors[k].kneighbors(des)

        template_pts = []
        img_pts = []

        dists = []

        for i, point in enumerate(kp):
            distance = distances[i][0]
            index = indices[i][0]

            if distance < self.good_thresh:
                dists.append(distance)
                img_pts.append(point.pt)

                template_pts.append(self.kps[k][index].pt)

                # print 'Match found with distance {}'.format(distance)

        print '{} points matched with median dist {}'.format(len(img_pts), np.median(dists))

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(np.array(img_pts), np.array(template_pts), cv2.RANSAC, self.ransac_thresh)
        self.img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        # TODO: change img to img_T once you do the homography transform
        visual_diff = compare_images(self.img_T, self.signs[k])
        return visual_diff


# end of TemplateMatcher class

def compare_images(img1, img2):

    cv2.imshow('match', img1)
    cv2.imshow('template', img2)

    cv2.waitKey(5)


    raw_input('Press enter to continue')

    return 0


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
