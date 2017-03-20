import cv2
import numpy as np

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {} # maps keys to the template images
        self.keypointsMap = {} # maps keys to the keypoints of the template images
        self.descriptorsMap = {} # maps keys to the descriptors of the template images
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold
        self.ransac_thresh = 5.0

        self.bf = cv2.BFMatcher()

        # Precompute keypoints for template images
        for key, filename in images.iteritems():

            # load template sign images as grayscale
            self.signs[key] = cv2.imread(filename, 0)

            # precompute keypoints and descriptors for the template sign
            self.keypointsMap[key], self.descriptorsMap[key] = self.sift.detectAndCompute(self.signs[key], None)

    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        # Get keypoints and descriptors from input image using SIFT
        # Store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img, None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        template_confidence = {k: 0 for k in self.signs.keys()}

        if visual_diff:
            normalizing_scale = sum(np.divide(1, visual_diff.values()))
            for key in visual_diff:
                template_confidence[key] = 1 / (visual_diff[key] * normalizing_scale)

        return template_confidence

    def _compute_prediction(self, k, img, keypoints, descriptors):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        keypoints: keypoints from scene image,   descriptors: descriptors from scene image
        """

        # Find corresponding points in the input image and the template image
        # put keypoints from template image in template_pts
        # put corresponding keypoints from input image in img_pts

        # http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
        matches = self.bf.knnMatch(self.descriptorsMap[k], descriptors, k = 2)

        # Apply ratio test
        good_matches = [m for (m, n) in matches if (m.distance < 0.75 * n.distance)]

        img_pts = np.asarray([keypoints[match.trainIdx].pt for match in good_matches])
        template_pts = np.asarray([self.keypointsMap[k][match.queryIdx].pt for match in good_matches])

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)

        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff

def compare_images(img1, img2):

    # Normalize
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)

    return np.linalg.norm(np.subtract(img1_norm, img2_norm))

if __name__ == '__main__':

    images = {
        "left": 'images/leftturn_box_small.png',
        "right": 'images/rightturn_box_small.png',
        "uturn": 'images/uturn_box_small.png'
    }

    tm = TemplateMatcher(images)

    scenes = [
        "images/uturn_scene.jpg",
        "images/leftturn_scene.jpg",
        "images/rightturn_scene.jpg"
    ]

    for filename in scenes:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)
        print filename.split('/')[-1]
        print pred
