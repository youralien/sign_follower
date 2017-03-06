#!/usr/bin/env python
import cv2
import numpy as np

"""
This code determines which of a set of template images matches
an input image the best using the ORB algorithm
"""

class TemplateMatcher(object):

    def __init__ (self, images, min_match_count=10, good_thresh=0.7):
        self.signs = {} #maps keys to the template images
        self.kps = {} #maps keys to the keypoints of the template images
        self.descs = {} #maps keys to the descriptors of the template images
        if cv2.__version__=='3.1.0-dev':
            self.orb = cv2.xfeatures2d.SIFT_create()
        else:
            self.orb = cv2.ORB() #initialize ORB to be used for image matching

        # for potential tweaking
        self.min_match_count = min_match_count
        self.good_thresh = good_thresh #use for keypoint threshold
        self.ransac_thresh = 5

        #TODO: precompute keypoints for template images

        for k, filename in images.iteritems():
            # load template sign images as grayscale
            self.signs[k] = cv2.imread(filename,0)

            # precompute keypoints and descriptors for the template sign 
            self.kps[k], self.descs[k] = self.orb.detectAndCompute(self.signs[k],None)


    def predict(self, img):
        """
        Uses gather predictions to get visual diffs of the image to each template
        returns a dictionary, keys being signs, values being confidences
        """
        visual_diff = {}

        for k in self.signs.keys():
            #cycle trough templage images (k) and get the image differences
            visual_diff[k] = self._compute_prediction(k, img, self.kps, self.descs)

        if visual_diff:
            diff_range = np.sum(visual_diff.values()) - min(visual_diff.values())
            template_confidence = {k: (visual_diff[k] - min(visual_diff.values()))/diff_range for k in self.signs.keys()}

            # TODO: convert difference between images (from visual_diff)
            #       to confidence values (stored in template_confidence)

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
        img_kp, img_des = self.orb.detectAndCompute(img,None)

        template_pts = np.array([])
        img_pts = np.array([])
        
        for i in xrange(len(des[k])):
            descriptor = des[k][i]
            closest = [False, False]
            second_closest = [False, False]
            for j in xrange(len(img_des)):
                img_descriptor = img_des[j]
                dist = np.linalg.norm(descriptor - img_descriptor, 2)
                if closest[0]:
                    if dist < closest[1]:
                        second_closest[0] = closest[0]
                        second_closest[1] = closest[1]
                        closest[0] = j
                        closest[1] = dist
                    elif (not second_closest[0]) or (dist < second_closest[1]):
                        second_closest[0] = j
                        second_closest[1] = dist
                else:
                    closest[0] = j
                    closest[1] = dist
            if closest[1] < (0.7*second_closest[1]):
                np.append(template_pts, [kp[k][i]])
                np.append(img_pts, [img_kp[i]])

        if len(img_pts) > 3:
            M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
            img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
            visual_diff = compare_images(img_T, self.signs[k])
            return visual_diff
        else:
            return compare_images(img, self.signs[k])
# end of TemplateMatcher class

def compare_images(img1, img2):
    img1_mean = np.sum(img1)/float(len(img1)*len(img1[0]))
    img2_mean = np.sum(img2)/float(len(img2)*len(img2[0]))
    img1 = (img1 - img1_mean)/np.std(img1)
    img2 = (img2 - img2_mean)/np.std(img2)

    img1 = cv2.resize(img1, (len(img2[0]), len(img2)))

    diff = img1 - img2
    diff_mean = np.sum(diff)/float(len(diff)*len(diff[0]))

    return 1.0/diff_mean

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