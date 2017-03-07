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

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold

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

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        template_confidence = {}   
        if visual_diff:
            for d in visual_diff.keys():
                template_confidence[d] = visual_diff[d]

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

        '''feature matching, source: 
        http://opencv-python-tutroals.readthedocs.io
        /en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html'''
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des,self.descs[k],k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # if there is more than 10 matches, extract the locations of matched keypoints in both the images
        if len(good) > self.min_match_count:
            img_pts = np.float32([kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            template_pts = np.float32([self.kps[k][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            # Transform input image so that it matches the template image as well as possible
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.good_thresh)
            img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
            visual_diff = compare_images(img_T, self.signs[k])
            return visual_diff
        
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            return None
# end of TemplateMatcher class

def compare_images(img1, img2):
    # find the corelation coefficient of the two images
    # first converts the image to 1-d
    img1 = np.ravel((img1-np.mean(img1))/np.std(img1))
    img2 = np.ravel((img2-np.mean(img2))/np.std(img2))
    # then get the corelation coeficient
    return np.corrcoef(img1,img2)[0,1]

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