import cv2

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

        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold        
        self.ransac_thres = 0.8 # FIXME: value?

        # precompute keypoints for template images
        for k, filename in images.iteritems():
            # load template sign images as grayscale

            # FIXME: cv2.imread returns None for these images
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
            visual_diff[k] = 1.2 # TODO: replace with self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # convert difference between images (from visual_diff)
            # to confidence values (stored in template_confidence)
            template_confidence_sum = 0.0
            for k in visual_diff:
                template_confidence[k] = 1/visual_diff[k]
                template_confidence_sum += template_confidence[k]
                print "template confidence", template_confidence[k]
            
            # normalize confidence values
            for k in template_confidence:
                template_confidence[k] /= template_confidence_sum

        else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}
            
        print "template_confidence_sum", template_confidence_sum

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

        # kp: input keypoints
        # self.kps: template keypoints

        # img_pts = 
        # template_pts = 

        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    # normalize images: (each_element - mean)/standard_dev
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)

    diff = np.substract(img1, img2) # absolute value?
    # matrix norm of difference
    norm_diff = np.linalg.norm(diff)
    return norm_diff