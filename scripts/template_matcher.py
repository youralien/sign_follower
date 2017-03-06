import cv2
import math
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
        # self.ransac_thresh
        if cv2.__version__=='3.1.0-dev':
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = cv2.SIFT() #initialize SIFT to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.ransac_thresh = good_thresh #use for keypoint threshold
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
        kp = None
        des = None
        

        kp, des = self.sift.detectAndCompute(img,None)



        for k in self.signs.keys():
            #cycle trough template images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, kp, des)

        if visual_diff:
            # print self.signs.keys()
            temp = []
            for k in self.signs.keys():
                temp.append(visual_diff[k])

            total = sum(temp)
            #The highest value is the most confident answer
            template_confidence = {k: 1-(visual_diff[k]/total) for k in self.signs.keys()}

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
        #For every key point on template, compare it to all input key points and pick the closest to match with


        template_pts = []
        img_pts = []

        for temp in self.kps[k]:
            #Improbably high distance
            lowest_dist = 1000000000
            curr_temp = None
            curr_scene = None
            #Calculate the distance between keypoints and take the best for each template keypoint
            for keyp in kp:
                x1 = temp.pt[0]
                x2 = keyp.pt[0]
                y1 = temp.pt[1]
                y2 = keyp.pt[1]
                distance = math.sqrt(((x2-x1)**2)+(y2-y1)**2)
                if distance < lowest_dist:
                    curr_temp = temp
                    curr_scene = keyp

            #Append coordinates
            template_pts.append([curr_temp.pt[0],curr_temp.pt[1]])
            img_pts.append([curr_scene.pt[0],curr_scene.pt[1]])

        template_pts = np.array(template_pts)
        img_pts = np.array(img_pts)
        # Transform input image so that it matches the template image as well as possible
        M, mask = cv2.findHomography(img_pts[:10], template_pts[:10], cv2.RANSAC, self.ransac_thresh)
        img_t = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])

        visual_diff = compare_images(img_t, self.signs[k])
        return visual_diff
# end of TemplateMatcher class

def compare_images(img1, img2):
    """Uses mean squared error to determine how similar the images are, given two images as inputs"""
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err



if __name__ == '__main__':
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }

    tm = TemplateMatcher(images)
    scenes = [
    "../images/uturn_box.png",
    "../images/leftturn_box.png",
    "../images/rightturn_scene.jpg"]

for filename in scenes:
    scene_img = cv2.imread(filename, 0)
    pred = tm.predict(scene_img)
    print filename.split('/')[-1]
    print pred