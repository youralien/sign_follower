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
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        self.bf = cv2.BFMatcher()
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold        

        # precompute keypoints for template images
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

        # get keypoints and descriptors from input image using SIFT
        # store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img, None)

        for k in self.signs.keys():
            #cycle through template images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # convert difference between images (from visual_diff)
            # to confidence values (stored in template_confidence)
            template_confidence_sum = 0.0
            for k in visual_diff:
                template_confidence[k] = 1/visual_diff[k]
                template_confidence_sum += template_confidence[k]
                # print "template confidence", template_confidence[k]
            
            # normalize confidence values
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

        # Sources:
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

        matches = self.bf.knnMatch(self.descs[k], des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        img_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        template_pts = np.float32([ self.kps[k][m.queryIdx].pt for m in good ]).reshape(-1,1,2)

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        visual_diff = compare_images(img_T, self.signs[k])

        return visual_diff
        
# end of TemplateMatcher class

def compare_images(img1, img2):
    # normalize images to remove lighting effects: (each_element - mean)/standard_dev
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)

    diff = np.subtract(img1, img2)
    # matrix norm of difference
    norm_diff = np.linalg.norm(diff)
    return norm_diff