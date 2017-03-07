import cv2
import numpy as np

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        """ Store template images, create the sift and matching objects
            which will be used for processing images

            images: dictionary of "template name":"template filename" pairs,
                which is used to find and store the actual images
            min_match_count: Designed to be the minimum number of keypoint
                matches for a template to be considered a possible
                match
            good_thresh: Used for homography transformation - determines whether
                or not keypoints will be kept in transformation
        """
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
        self.matcher = cv2.BFMatcher()

        self.get_template_keypoints(images)

    def get_template_keypoints(self, template_imgs):
        """ Finds keypoints in the template images, which means that you
            don't have to constantly recompute them.

            template_imgs: dictionary of "template name":"template filename" pairs
        """
        for k, filename in template_imgs.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)

            # precompute keypoints and descriptors for the template sign
            self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)



    def predict(self, img):
        """ Uses gather predictions to get visual diffs of the image to each template
            returns a dictionary, keys being signs, values being confidences

            img: opencv image, to be matched to the templates
        """
        visual_diff = {}
        #Find the keypoints for the images
        kp, des = self.sift.detectAndCompute(img,None)

        for k in self.signs.keys():
            #cycle through templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            #confidence will now be based on ratio of visual differences
            #and will sum to 1.
            total = sum([visual_diff[key] for key in visual_diff.keys()])
            template_confidence = {key: visual_diff[key]/total for key in self.signs.keys()}

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

        # Get list of matches between keypoints based on destriptors
        matches = self.matcher.knnMatch(self.descs[k],des,k=2)
        # Convert into point form
        matches = [(m.queryIdx, m.trainIdx) for m, n in matches]

        # Convert those points to points findHomography can handle
        template_pts = np.zeros((len(matches),2))
        img_pts = np.zeros((len(matches),2))
        for idx in range(len(matches)):
            match = matches[idx]
            template_pts[idx,:] = self.kps[k][match[0]].pt
            img_pts[idx,:] = kp[match[1]].pt

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.good_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        #Find total differences
        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    #normalize the images
    norm1 = (img1 - np.mean(img1))/np.std(img1)
    norm2 = (img2 - np.mean(img2))/np.std(img2)

    #computing root mean squared error between the two arrays
    #Average error between normalized pixels
    thing = np.sqrt(((norm1 - norm2) ** 2).mean())

    #A perfect match is zero - the worse that match, the larger
    #the error. So, invert it.
    return 1/thing

if __name__ == '__main__':
    images = {
        "uturn": '../images/uturn_box_small.png',
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png'
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
        print '\n'
