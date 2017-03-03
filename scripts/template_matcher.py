import cv2
import numpy as np
import math
import sys
import time

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images

        self.sift = cv2.xfeatures2d.SIFT_create()

        self.img_T = None


        #TODO: what do these do?
        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold
        self.ransac_thresh = 5

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

        kp, des = self.sift.detectAndCompute(img, None)
        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            difference = self._compute_prediction(k, img, kp, des)
            if difference is None:
                difference = np.inf
            visual_diff[k] = difference

        if visual_diff:
            # for k in visual_diff.keys():
            #     print visual_diff[k]

            template_confidence = {k:1/visual_diff[k] for k in visual_diff.keys()}
            temp_sum = np.sum(template_confidence.values())
            for k in template_confidence.keys():
                template_confidence[k] /= temp_sum

        else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}

        # signs = [(k, template_confidence[k]) for k in self.signs.keys() if template_confidence[k] > self.good_thresh]
        # sign = sorted(signs, key=lambda x: x[1], reverse=True)
        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        """

        template_pts = []
        img_pts = []
        similarity = sys.maxint
        current_kp = None
        for i in range(len(self.kps[k])):
            for j in range(len(kp)):
                temp = np.sum(abs(des[j] - self.descs[k][i]))
                if(similarity > temp):
                    similarity = temp
                    current_kp = kp[j]

            if(similarity != sys.maxint):
                template_pts.append(self.kps[k][i].pt)
                img_pts.append(current_kp.pt)
                similarity = sys.maxint
        if img_pts >= self.min_match_count:
            img_pts = np.asarray(img_pts, dtype=float)
            template_pts = np.asarray(template_pts, dtype=float)
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
            self.img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
            # cv2.imshow("img_T", tm.img_T)
            # cv2.waitKey(5)
            # time.sleep(5)
            visual_diff = compare_images(self.img_T, self.signs[k])
            return visual_diff
        else:
            return None

# end of TemplateMatcher class

def compare_images(img1, img2):
    img1_mean = np.mean(img1)
    img1_std = np.std(img1)
    img2_mean = np.mean(img2)
    img2_std=  np.std(img2)
    norm_img1 = (img1 - img1_mean)/img1_std
    norm_img2 = (img2 - img2_mean)/img2_std

    return np.sum(abs(norm_img2 - norm_img1))





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
    # cv2.namedWindow("img_T")
    for filename in scenes:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)

        print filename.split('/')[-1]
        print pred
        print ' '
