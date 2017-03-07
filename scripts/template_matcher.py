import cv2, math
import numpy as np

TEMPLATES = {
    "left": '../images/leftturn_box_small.png',
    "right": '../images/rightturn_box_small.png',
    "uturn": '../images/uturn_box_small.png',
}

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images=TEMPLATES, debug=False, min_match_count=10, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        if cv2.__version__ =='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        if debug:
            cv2.namedWindow('base')
            cv2.namedWindow('transformed')
            cv2.namedWindow('template')

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold
        self.ransac_thresh = 5.0
        self.debug = debug

        for k, filename in images.items():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename, 0)

            # load template sign images as grayscale
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k], None)


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        kp, des = self.sift.detectAndCompute(img, None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            template_confidence = {k: math.e**-(v) for k, v in visual_diff.items()}

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

        # Create a brute-force matcher which selects the best matches between
        # keypoint sets. Then filter down to only the top 10 best (least-distance)
        # keypoints.
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(self.descs[k], des)
        matches = sorted(matches, key = lambda x: x.distance)[:10]

        # Extract and format template and image points for input to findHomography
        img_pts = np.float32([m.trainIdx for m in matches]).reshape(-1,1,2)
        template_pts = np.float32([m.queryIdx for m in matches]).reshape(-1,1,2)

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        if self.debug:
            cv2.imshow('base', img)
            cv2.imshow('transformed', img_T)
            cv2.imshow('template', self.signs[k])
            cv2.waitKey()

        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    return np.mean(np.abs(np.subtract(img1, img2)))

def normalize(img):
    m, s = (np.mean(img), np.std(img))
    for elem in np.nditer(img, op_flags=['readwrite']):
        elem[...] = (elem-m)/s

    return img

if __name__ == "__main__":
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png',
    }

    tm = TemplateMatcher(images, debug=True)

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