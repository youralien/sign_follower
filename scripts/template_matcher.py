#!/usr/bin/env python 

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

import cv2
import numpy as np

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.5):
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

        self.img_T = None
        self.ransac_thresh = 5.0

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
        
        # compute keypoint and descriptor for img
        kp, des = self.sift.detectAndCompute(img,None)

        if (des != None and img != None):
            for k in self.signs.keys():
                #cycle trough templage images (k) and get the image differences
                visual_diff[k] = self._compute_prediction(k, img, kp, des)

        # Find template_confidence by inverting and normalizing visual_diff
        if visual_diff:
            template_confidence = visual_diff
            total = sum(template_confidence.values())
            for k in template_confidence:
                if (template_confidence[k] != 0):
                    template_confidence[k] /= total

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

        img_pts = []
        template_pts = []
        closest_dists = []

        # Quantify the similarity between each descriptor in the streamed image
        # and each descriptor in the specified template image. Only keep track
        # of the two closest template descriptor matches for each stream image
        # descriptor.
        for img_des_num, img_des in enumerate(des):
            closest_dists.append([
                {"temp_des_num":0, "dist":1000000},
                {"temp_des_num":0, "dist":999999}
            ])
            for temp_des_num, temp_des in enumerate(self.descs[k]):
                dist = np.linalg.norm(img_des-temp_des, 2)
                # Keep dist if closer than one of closest distances
                if (dist < closest_dists[img_des_num][0]["dist"]):
                    if (dist < closest_dists[img_des_num][1]["dist"]):
                        closest_dists[img_des_num][0] = \
                            closest_dists[img_des_num][1]
                        closest_dists[img_des_num][1] = \
                            {"temp_des_num":temp_des_num, "dist":dist}
                    else:
                        closest_dists[img_des_num][0] = \
                            {"temp_des_num":temp_des_num, "dist":dist}

        # Go through the collected closest distances between stream image
        # descriptors and template image descriptors. If any match passes
        # the good threshold, then add it to the appropriate list.
        for img_des_num, close_dist in enumerate(closest_dists):
            if (close_dist[1]["dist"] < self.good_thresh*close_dist[0]["dist"]):
                img_pts.append(kp[img_des_num].pt)
                template_pts.append(self.kps[k][close_dist[1]["temp_des_num"]].pt)

        img_pts = np.asarray(img_pts)
        template_pts = np.asarray(template_pts)
                
        # Transform input image so that it matches the template image as well as possible
        if (len(img_pts) > 3):
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
            self.img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

            visual_diff = compare_images(self.img_T, self.signs[k])
        else:
            # Image could not be warped, return a a large visual_diff
            visual_diff = 100000000000

        return visual_diff


def compare_images(img1, img2):
    """
    Return value for how similar two images are after being normalized. The
    returned value will be between 0 and 255, where 0 means the images are
    identical.
    img1: first image to compare
    img2: second image to compare
    """

    # Find mean and standard deviation of each image
    img1_mean = np.mean(img1)
    img1_std = np.std(img1)
    img2_mean = np.mean(img2)
    img2_std = np.std(img2)

    # Normalize the images to remove lighting differences
    img1 = img1 - img1_mean / img1_std
    img2 = img2 - img2_mean / img2_std

    # Find the absolute difference between the two images
    diff = abs(img1-img2)
    diff_mean = np.mean(diff)

    return diff_mean

if __name__ == '__main__':
    images = {
        "left":'../images/leftturn_box_small.png',
        "right":'../images/rightturn_box_small.png',
        "uturn":'../images/uturn_box_small.png'
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
