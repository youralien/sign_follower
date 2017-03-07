import cv2
import numpy as np
import sys
import math

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm

Predictions are not entirely accurate at this time.

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
        template_confidence = {}

        if img != None:

            # kp is a list of keypoints
            # des is a numpy array of shape number of keypoints x 128
            # http://docs.opencv.org/3.2.0/da/df5/tutorial_py_sift_intro.html
            kp, des = self.sift.detectAndCompute(img, None)
            for k in self.signs.keys():
                #cycle through templage images (k) and get the image differences
                visual_diff[k] = self._compute_prediction(k, img, kp, des)
        # If any visual_diff was computed        
        
        if visual_diff:
            # Calculate confidences
            for k in visual_diff:
                template_confidence[k] = 1/visual_diff[k] # small diffs become big confidences, big diffs become small confidences
            template_confidence_sum = sum(template_confidence.values())

            # Normalize confidences
            for k in template_confidence:
                template_confidence[k] /= template_confidence_sum 

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

        # find corresponding points in the input image and the template image
        # put keypoints from template image in template_pts
        # put corresponding keypoints from input image in img_pts

        # USED THIS TUTORIAL
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.descs[k], des, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < self.good_thresh*n.distance:
                good.append(m)

        if len(good)>self.min_match_count:
            # self.kps are keypoints of template image
            # kp are keypoints from scene image
            img_pts = np.float32([ self.kps[k][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            template_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
        else:
            # Not enough matches found.  
            return 10000000000

        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff

# end of TemplateMatcher class

def compare_images(img1, img2):
    mean1 = np.average(img1) # mean of image 1 for normalization
    mean2 = np.average(img2) # mean of image 2 for normalization
    standard_dev1 = np.std(img1) # standard deviation of image 1 for normalization
    standard_dev2 = np.std(img2) # standard deviation of image 2 for normalization

    # Cannot divide by 0
    if standard_dev1 == 0:
        standard_dev1 = 0.000001 
    if standard_dev2 == 0:
        standard_dev2 = 0.000001 

    # Normalize
    img1_norm = img1 - mean1/standard_dev1
    img2_norm = img2 - mean2/standard_dev2

    # Differences 
    diff = abs(img1_norm - img2_norm) # np array of differences
    return np.mean(diff) # take mean of diffs and return


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