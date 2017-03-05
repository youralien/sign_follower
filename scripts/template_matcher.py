import cv2
import numpy as np
import time

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, ransac_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.ransac_thresh = ransac_thresh #use for keypoint threshold

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

        # get keypoints and descriptors from input image using SIFT
        # store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img,None)

        # iterate through signs.keys to find probability of each key
        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            sum_diffs = sum([visual_diff[k] for k in self.signs.keys()])
            template_confidence = {k: visual_diff[k]/sum_diffs for k in self.signs.keys()}

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

        # TODO: find corresponding points in the input image and the templae image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        # Transform input image so that it matches the template image as well as possible

        # make flan matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find the matches between known sign and img
        matches = flann.knnMatch(self.descs[k], des,k=2)

        # make list of good matches
        good = [m for m,n in matches if m.distance < 0.7*n.distance]

        src_pts = np.float32([ self.kps[k][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # if len(good)>MIN_MATCH_COUNT:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        #TODO: change img to img_T once you do the homography transform
        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    img1 = (img1 - img1.mean())/img1.std()
    img2 = (img2 - img2.mean())/img2.std()

    return 1 / (np.absolute(img1-img2).sum() + 1)

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
