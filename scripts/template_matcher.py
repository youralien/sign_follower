import cv2
import numpy as np
import math

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

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold

        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)

            # precompute keypoints and descriptors for the template sign
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)

        self.ransac_thresh = 3
        cv2.namedWindow('original')


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        # TODO: get keypoints and descriptors from input image using SIFT
        #       store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img, None)
        template_confidence = []

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # TODO: convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)
            template_confidence.append(visual_diff)

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

        # TODO: find corresponding points in the input image and the template_confidence image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des, self.descs[k], 2)
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        img_pts = []
        template_pts = []
        for i in good:
            template_pts.append(self.kps[k][i.trainIdx].pt)
            img_pts.append(kp[i.queryIdx].pt)

        
        M, mask = cv2.findHomography(np.float32(img_pts), np.float32(template_pts), cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
        cv2.imshow('original', img_T)
        visual_diff = compare_images(img_T, self.signs[k])




        # M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        # img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
        # #TODO: change img to img_T once you do the homography transform
        # visual_diff = compare_images(img_T, self.signs[k])

        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    height, width, channels = img1.shape
    count = 0
    print img1[1, 1]
    for i in range(height):
        for j in range(width):
            if math.fabs(img1[i, j] - img2[i, j]) < 5:
                count += 1
    return count


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
    cv2.waitKey(0)
