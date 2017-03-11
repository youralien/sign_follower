import cv2
import numpy as np
from scipy.spatial.distance import cdist

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, min_match_count=10, good_thresh=0.7):
        self.signs = {'left':'../images/leftturn_box_small.png','right':'../images/rightturn_box_small.png','uturn':'../images/uturn_box_small.png'} #maps keys to the template images
        self.im = {} # dictionary maps keys to image files read in open.cv
        self.im_bw = {} # dictionary maps keys to bw image files read in open.cv
        self.kps = {} # dictionary maps keys to the keypoints of the template images
        self.descs = {} # dictionary maps keys to the descriptors of the template images
        self.ransac_thresh = 5 # look up documentation
        self.score = {} # dictionary of template confidences

        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold

        #TODO: precompute keypoints for template images
        for k, filename in self.signs.items():
            self.im[k] = cv2.imread(filename)
            self.im_bw[k] = cv2.cvtColor(self.im[k],cv2.COLOR_BGR2GRAY)
            self.kps[k] = self.sift.detect(self.im_bw[k])
            _,self.descs[k] = self.sift.compute(self.im_bw[k],self.kps[k])

        # print([len(self.kps[k]) for k in self.signs])
        # cv2.namedWindow('image_window')

    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        # TODO: get keypoints and descriptors from input image using SIFT
        #       store keypoints in variable kp and descriptors in des

        kp = self.sift.detect(img)
        _,des = self.sift.compute(img,kp)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # print visual_diff
            # TODO: convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)
            diff = 0.0
            for k in self.signs.keys():
                visual_diff[k] = 1/visual_diff[k]
                diff += visual_diff[k]
            for k in self.signs.keys():
                self.score[k] = visual_diff[k]/diff
            template_confidence = {k: self.score[k] for k in self.signs.keys()}

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

        # TODO: find corresponding points in the input image and the template image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        self.template_pts = []
        self.img_pts = []

        distances_matrix = cdist(self.descs[k],des)
        # closest vector in des2 to the 0th descriptor vector in des1
        closest = distances_matrix.argsort()[:,:2]
        # print closest.shape
        for i in range(len(self.descs[k])):
            # print distances_matrix[i, closest[0], self.good_thresh*distances_matrix[i,closest[1]], closest[0]]
            if distances_matrix[i, closest[i,0]] < self.good_thresh*distances_matrix[i,closest[i,1]]:
                self.template_pts.append(self.kps[k][i].pt)
                self.img_pts.append(kp[closest[i,0]].pt)
        self.img_pts = np.array(self.img_pts)
        self.template_pts = np.array(self.template_pts)
        # print len(self.template_pts), type(self.img_pts), type(self.template_pts)

        #TODO: change img to img_T once you do the homography transform
        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(self.img_pts, self.template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.im_bw[k].shape[::-1])
        # cv2.imshow('image_window',img_T)
        # cv2.waitKey(0)
        visual_diff = compare_images(img_T, self.im_bw[k])
        return visual_diff

# end of TemplateMatcher class

def compare_images(img1, img2):
    # return 0
    norms1 = []
    norms2 = []
    visual_diff = 0
    for each_element in range(len(img1)):
        norm1 = (each_element - np.average(img1))/np.std(img1)
        norms1.append(norm1)
    # print norms1

    for each_element in range(len(img2)):
        norm2 = (each_element - np.average(img2))/np.std(img2)
        norms2.append(norm2)
    # print norms2

    for i in range(len(norms1)):
        visual_diff += abs(norms1[i]-norms2[i])
    print visual_diff
    return visual_diff



if __name__ == '__main__':
    # import os
    node = TemplateMatcher()
    # imgpath = '../images/leftturn_scene.jpg'
    # print "Does image exist: ", os.path.exists(imgpath)
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    print node.predict(img)