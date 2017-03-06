from __future__ import division
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
        self.ransac_thresh = 5.0 #found experimentally to work well


        #windows for diagnostic imgs and transforms
        # cv2.namedWindow('img1_window')
        # cv2.namedWindow('img2_window')
        # cv2.namedWindow('difference_window')

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

        #get keypoints and descriptors from image using sift
        kp,des = self.sift.detectAndCompute(img,None)


        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            #invert values
            for k in visual_diff:
                visual_diff[k] = 1/visual_diff[k]
            #normalize
            for k in visual_diff:
                template_confidence[k] = visual_diff[k]/sum(visual_diff.values())

        else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            template_confidence = {k: 0 for k in self.signs.keys()}
            


        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        CV2 Brute Force Matching implemented as per this tutorial: 
        http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        """
        #brute force match points
        bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
        #generate matched array
        matched = bf.match(self.descs[k],des)
        #sort in order of difference
        matched = sorted(matched, key = lambda x:x.distance)


        kp1_match = [kp[match.trainIdx] for match in matched]
        kp2_match = [self.kps[k][match.queryIdx] for match in matched]

        #create keypoint array
        img_pts = np.array([keypoint.pt for keypoint in kp1_match])
        template_pts = np.array([keypoint.pt for keypoint in kp2_match])


        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
        img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
        visual_diff = compare_images(img_T, self.signs[k])
        return visual_diff

# end of TemplateMatcher class

def compare_images(img1, img2):
    """Compares two given images and returns a single value corresponding to their difference"""
    

    #normalize imagesfor brightness correction
    

    img1 =(img1-np.mean(img1))/np.std(img1)
    img2 =(img2-np.mean(img2))/np.std(img2)

    # cv2.imshow('img1_window',img1)
    # cv2.imshow('img2_window',img2)
    # cv2.waitKey(3000)

    #compute difference matrix
    difference = img1 - img2

    #diagnostic show images and transforms

    #cv2.imshow('difference_window',difference)
   
    
    #compute norm
    norm = np.linalg.norm(difference)
    return norm



if __name__ == '__main__':
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }

    tm = TemplateMatcher(images)


    scenes = [
    "../images/leftturn_scene.jpg",
    "../images/rightturn_scene.jpg",
    "../images/uturn_scene.jpg"
    ]

    for filename in scenes:
        scene_img = cv2.imread(filename, 0)
        pred = tm.predict(scene_img)
        print filename.split('/')[-1]
        print pred