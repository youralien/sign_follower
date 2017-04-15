import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
This code determines which of a set of template images matches
an input image the best using the SIFT algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=8, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        self.matcher = cv2.BFMatcher() #initialize matcher for descriptors
        self.ransac_thresh = 5.0
        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold
        self.corner_threshold = 0.039

        #precompute keypoints for template images
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
        #       store keypoints in variable kp and descriptors in des
        kp, des = self.sift.detectAndCompute(img,None)

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)
            # get keys of minimum and maximum values

            # if there aren't enough good matches, None values will be returned
            # handle those cases where value could be None
            #extract version without None values
            res = {k:v for k,v in visual_diff.items() if v is not None}
            print res
            #in case all values are None
            template_confidence = {k: 0 for k in self.signs.keys()}
            if not res == {}:
                min_vd = min(res, key=res.get)
                max_vd = max(res, key=res.get)
            
                for k in self.signs.keys():
                    minimum = visual_diff[min_vd]
                    maximum = visual_diff[max_vd]
                    if minimum == maximum:
                        maximum += 1
                        minimum = 0
                    
                    if (visual_diff[k] == None):
                        template_confidence[k] = 0.0
                        print "visual_diff[k] is none" 
                    else: #compute confidence as value between 0 and 1 where 1 is confident and 0 is not
                        template_confidence[k] = 1.0 - ((visual_diff[k] - minimum) / (maximum - minimum))

        #else: # if visual diff was not computed (bad crop, homography could not be computed)
            # set 0 confidence for all signs
            #template_confidence = {k: 0 for k in self.signs.keys()}
            #print "else for template_confidence[k]"

        return template_confidence


    def _compute_prediction(self, k, img, kp, des):
        """
        Return comparison values between a template k and given image
        k: template image for comparison, img: scene image
        kp: keypoints from scene image,   des: descriptors from scene image
        """

        # find corresponding points in the input image and the templae image
        #       put keypoints from template image in template_pts
        #       put corresponding keypoints from input image in img_pts
        matches = self.matcher.knnMatch(des,self.descs[k],k=2)

        good_matches = []
        for m,n in matches:
            # make sure the distance to the closest match is sufficiently better than the second closest
            if (m.distance < self.good_thresh*n.distance):  
                # and
                # kp[m.queryIdx].response > self.corner_threshold and
                # self.kps[k][m.trainIdx].response > self.corner_threshold):
                good_matches.append(m)

        # Draw first good matches.
        img3 = cv2.drawMatches(img,kp,self.signs[k],self.kps[k],good_matches, None, flags=2)
        plt.imshow(img3),plt.show()

        print len(good_matches)

        if len(good_matches)>self.min_match_count:
            #print "it is!"
            img_pts = np.float32([ kp[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            template_pts = np.float32([ self.kps[k][m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            
            #set in case findHomography doesn't compute correctly sized matrix for warpPerspective
            # visual_diff = 1000

            # Transform input image so that it matches the template image as well as possible
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
            # if(np.size(M) == 9):
            img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
            plt.imshow(img_T),plt.show()

            visual_diff = compare_images(img_T, self.signs[k])
            return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    #normalize images by: (each_element - mean)/standard_dev
    img1_n = (img1 - np.mean(img1)) / np.std(img1)
    img2_n = (img2 - np.mean(img2)) / np.std(img2)

    #return the sum of the absolute differences as comparison
    return np.sum(np.fabs(img1_n-img2_n)) 

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