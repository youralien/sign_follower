# Sign Detector
This project, based on one by [Ryan Louie](https://github.com/youralien), Kai Levy, and Dakota Nelson, was provided as a scaffolded way to learn the basics of CV in ROS by the [Computational Robotics](https://sites.google.com/site/comprobo17/) class at Olin College of Engineering. In this project, images from a live camera on a Neato robot are filtered, cropped, fed through image recognition that...

## System

### Detecting Signs in Scene Images
Reliable detection of traffic signs and creating accurate bounding box crops is an important preprocessing step for further steps in the data pipeline.

You will implement code that will find the bounding box around the traffic sign in a scene. We've outlined a suggested data processing pipeline for this task.

1. colorspace conversion to hue, saturation, value (```hsv_image``` seen in the top left window).
2. a filter is applied that selects only for objects in the yellow color spectrum. The range of this spectrum can be found using hand tuning (```binarized_image``` seen in the bottom window).
3. a bounding box is drawn around the most dense, yellow regions of the image.

![][yellow_sign_detector]
[yellow_sign_detector]: images/yellow-sign-detector.gif "Bounding box generated around the yellow parts of the image.  The video is converted to HSV colorspace, an inRange operation is performed to filter out any non yellow objects, and finally a bounding box is generated."

You will be writing all your image processing pipeline within the `process_image` callback function. Here is what the starter code looks like so far.

```python
    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        left_top, right_bottom = self.sign_bounding_box()
        left, top = left_top
        right, bottom = right_bottom

        # crop bounding box region of interest
        cropped_sign = self.cv_image[top:bottom, left:right]

        # draw bounding box rectangle
        cv2.rectangle(self.cv_image, left_top, right_bottom, color=(0, 0, 255), thickness=5)
```

The goal of localizing the signs in the scene is to determine `left_top = (x1,y1)` and `right_bottom = (x2,y2)` points that define the upper lefthand corner and lower righthand corner of a bounding box around the sign. You can do most of your work in the instance method `sign_bounding_box`.

```python
    def sign_bounding_box(self):
        """
        Returns
        -------
        (left_top, right_bottom) where left_top and right_bottom are tuples of (x_pixel, y_pixel)
            defining topleft and bottomright corners of the bounding box
        """
        # TODO: YOUR SOLUTION HERE
        left_top = (200, 200)
        right_bottom = (400, 400)
        return left_top, right_bottom"
```

Whether you follow along with the suggested steps for creating a sign recognizer or have ideas of your own, revisit these questions often when designing your image processing pipeline:

* What are some distinguishing visual features about the sign?  Is there similarities in color and/or geometry?
* Since we are interested in generating a bounding box to be used in cropping out the sign from the original frame, what are different methods of generating candidate boxes?
* What defines a good bounding box crop?  It depends a lot on how robust the sign recognizer you have designed.

Finally, if you think that working with individual images, outside of the `StreetSignRecognizer` class would be helpful -- I often like to prototype the computer vision algorithms I am developing in a jupyter notebook -- feel free to use some of the image frames in the `images/` folder.  In addition, you can save your own images from the video feed by using OpenCV's [`imwrite` method](http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imwrite#imwrite).

#### Red-Green-Blue to Hue-Saturation-Value Images

There are different ways to represent the information in an image. A gray-scale image has `(n_rows, n_cols)`. An rgb image has shape `(n_rows, n_cols, 3)` since it has three channels: red, green, and blue (note: as you saw in class, and it is the case with the given starter code, that OpenCV uses the channel ordering blue, green, red instead).

Color images are also represented in different ways too.  Aside from the default RGB colorspace, there exists alot of others. We'll be focused on using [HSV/HSL](https://en.wikipedia.org/wiki/HSL_and_HSV): Hue, Saturation, and Value/Luminosity. Like RGB, a HSV image has three channels and is shape `(n_rows, n_cols, 3)`. The hue channel is well suited for color detection tasks, because we can filter by color on a single dimension of measurement, and it is a measure that is invariant to lighting conditions.

[OpenCV provides methods to convert images from one color space to another](http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor).

A good first step would be convert `self.cv_image` into an HSV image and visualize it. Like any good roboticist, visualize everything to make sure it meets your expectations.  Note: if you are using OpenCV 3.1 (which is the case for anyone on Kinetic and Ubuntu 16.04), make sure to never called cv2.imshow from one of your sensor callback threads.  You should only ever call it from the main thread.

#### Filtering the image for only yellow

Since the set of signs we are recognizing are all yellow, by design, we can handtune a filter that will only select the certain shade of yellow in our image.

Here's a callback that will help to display the RGB value when hovering over the image window with a mouse (Note: you get this behavior for free with OpenCV 3.1).

```python
    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values associated
            with a particular pixel in the camera images """
        self.image_info_window = 255*np.ones((500,500,3))

        # show hsv values
        cv2.putText(self.image_info_window,
                    'Color (h=%d,s=%d,v=%d)' % (self.hsv_image[y,x,0], self.hsv_image[y,x,1], self.hsv_image[y,x,2]),
                    (5,50), # 5 = x, 50 = y
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))

        # show bgr values
        cv2.putText(self.image_info_window,
                    'Color (b=%d,g=%d,r=%d)' % (self.cv_image[y,x,0], self.cv_image[y,x,1], self.cv_image[y,x,2]),
                    (5,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))
```

In the `__init__` method, connect this callback by adding the following line:

```python
self.image_info_window = None
cv2.setMouseCallback('video_window', self.process_mouse_event)
```

And add the following lines to your run loop:

```python
            if not self.image_info_window is None:
                cv2.imshow('image_info', self.image_info_window)
                cv2.waitKey(5)
```

Now, if you hover over a certain part of the image, it will tell you what R, G, B value you are hovering over. Once you have created an HSV image, you can edit this function to also display the Hue, Saturation, and Value numbers.

OpenCV windows can be pretty powerful when setting up interactive sliders to change parameters.  As stated in class for Neato soccer, if you want to learn dynamic_reconfigure, you can use that instead of OpenCV's trackbars.

In the `__init__` method, copy the following lines which
```python
            cv2.namedWindow('threshold_image')
            self.hsv_lb = np.array([0, 0, 0]) # hsv lower bound
            cv2.createTrackbar('H lb', 'threshold_image', 0, 255, self.set_h_lb)
            cv2.createTrackbar('S lb', 'threshold_image', 0, 255, self.set_s_lb)
            cv2.createTrackbar('V lb', 'threshold_image', 0, 255, self.set_v_lb)
            self.hsv_ub = np.array([255, 255, 255]) # hsv upper bound
            cv2.createTrackbar('H ub', 'threshold_image', 0, 255, self.set_h_ub)
            cv2.createTrackbar('S ub', 'threshold_image', 0, 255, self.set_s_ub)
            cv2.createTrackbar('V ub', 'threshold_image', 0, 255, self.set_v_ub)
```

Then, add the following callback methods to the class definition that respond to changes in the trackbar sliders

```python
    def set_h_lb(self, val):
        """ set hue lower bound """
        self.hsv_lb[0] = val

    def set_s_lb(self, val):
        """ set saturation lower bound """
        self.hsv_lb[1] = val

    def set_v_lb(self, val):
        """ set value lower bound """
        self.hsv_lb[2] = val

    def set_h_ub(self, val):
        """ set hue upper bound """
        self.hsv_ub[0] = val

    def set_s_ub(self, val):
        """ set saturation upper bound """
        self.hsv_ub[1] = val

    def set_v_ub(self, val):
        """ set value upper bound """
        self.hsv_ub[2] = val
```

The sliders will help set the hsv lower and upper bound limits (`self.hsv_lb` and `self.hsv_ub`), which you can then use as limits for filtering certain parts of the HSV spectrum. Check out the OpenCV [inRange method](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html) for more details on how to threshold an image for a range of a particular color.

By the end of this step, you should have a binary image mask where all the pixels that are white represent the color range that was specified in the thresholding operation.

#### Generating a bounding box

You can develop an algorithm that operates on the binary image mask that you developed in the step above.

One method that could be fruitful would be dividing the image in a grid.  You might want to write a method that divides the image into a binary grid of `grid_size=(M,N)`; if tile in the grid contains a large enough percentage of white pixels, the tile will be turned on.

Since the images are stored as 2D arrays, you can use NumPy-like syntax to slice the images in order to obtain these grid cells. We've provided an example in `grid_image.py` which I'll show here:

```python
import cv2
import os

imgpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../images/leftturn_scene.jpg")
img = cv2.imread(imgpath)

grid_cell_w = 64*3
grid_cell_h = 48*3

cv2.namedWindow("my_window")

# NumPy array slicing!!
grid_cell = img[grid_cell_h:2*grid_cell_h,
                grid_cell_w:2*grid_cell_w] 

cv2.imshow("my_window", grid_cell)
cv2.waitKey(0);
```

The task now is to decide which grid cells contain the region of interest. You can write another function that takes this binary grid and determines the bounding box that will include all the grid cells that were turned on.

![][grid]
[grid]: images/grid.png

OpenCV has a method called `boundingRect` which seems promising too.  I found a nice example, albeit in C++, that [finds a bounding rectangle using a threshold mask](http://answers.opencv.org/question/4183/what-is-the-best-way-to-find-bounding-box-for-binary-mask/) like you have.  It seems like it will depend on your thresholding operation to be pretty clean (i.e. no spurious white points, the only object that should be unmasked is the sign of interest).

![][boundingRectStars]
[boundingRectStars]: images/boundingRectStars.png

The goal is to produce `left_top = (x1, y1)` and `right_bottom = (x2, y2)` that define the top left and bottom right corners of the bounding box.

## Recognition

Recognizing the signs involves determining how well the cropped image if the sign matches the template image for each type of sign. To do this, we will find keypoints in the template image and in the input image, then see how well we can align the keypoints, and finally see how similar the aligned images are.

### testing

We have template images as well as static test images for the template matcher in the repository, so we can run the code with the correct template images and try to match them to the static test images we have.

To do this, first we need to initialize the template matcher with the template images:
``` python
if __name__ == '__main__':
    images = {
        "left": '../images/leftturn_box_small.png',
        "right": '../images/rightturn_box_small.png',
        "uturn": '../images/uturn_box_small.png'
        }
        
    tm = TemplateMatcher(images)
```
You can put this at the bottom of the file, and this if statement will mean that this part won't run when template_matcher is imported by other files.

Next, we can run `tm.predict` on our test scenes using this:
``` python
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
```
This reads the test images, runs `tm.predict` on each image, and prints the file name followed by the prediction. Given just the starter code, this should be the output:
``` bash
uturn_scene.jpg
{}
leftturn_scene.jpg
{}
rightturn_scene.jpg
{}
```

### finding keypoints

We are finding keypoints using open cv's implementation of the [SIFT algorithm](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html), then filtering the keypoints ourselves to find the points that match between the input image and each template.

For the template images, we can calculate the keypoints in the initialization function because the images won't chage. To find those keypoints, we can cycle through the input dictionary of template images, read the image files as grayscale images and compute the keypoints using openCV's SIFT implementation:
``` python
for k, filename in images.iteritems():
    # load template sign images as grayscale
    self.signs[k] = cv2.imread(filename,0)

    # precompute keypoints and descriptors for the template sign 
    self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)
```
The `predict` method is the "main" method of TemplateMatcher. It begins by finding the keypoints in the input image as the first step to matching it with at template. At this point, the template images are initialized and the predict method should run, so you can run the program and it should return predictions for each template image with a zero confidence value. Next, `predict` calls `_compute_prediction` to find how well the input image matches each template image and stores the predictions in a dictionary.

### aligning keypoints

In `_compute_prediction`, the first step to aligning the images is to find which keypoints from the two images match. The simplest method for that is to say keypoints whcih are close to eachother match. (note, later steps will correct for "matched" keypionts from mismatched images)

Based on the matched keypoints, the following lines find how to transform the input image so the keypoints align with the template images keypoints (using a homography matrix), then transorms the input image using that matrix, so it should align with the template.
``` python
# Transform input image so that it matches the template image as well as possible
M, mask = cv2.findHomography(img_pts, template_pts, cv2.RANSAC, self.ransac_thresh)
img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
```
Once you add these lines, you should change `img` in the line `visual_diff = compare_images(img, self.signs[k])` to `img_T`.

### comparing images

At this point, we have two images which, if they are of the same sign, should be aligned with each other. If they are of different signs, the matched keypoints were likely not well aligned, and the homography matrix probably skewed the image into an unrecognizable blob, but the computer can't tell what is a reasonable image and what is an unrecognizable blob, so now we have to determine how similar the two images are.

The `compare_images` function at the bottom of the file is used to find how similar two images are to each other. This one is left up to you, but here are a few hints:

First, there is one thing we have yet to account for while comparing images: lighting. If you have tried to do blob detection and then tried again when the sun went down, you know that lighting wreaks havoc on computer vision. Since we are using grayscale images, and we have cropped them so both images are of the same thing, we can eliminate most of the effects of lighting by normalizing the image. Mathematically this can be done by taking `(each_element - mean)/standard_dev`. Images are stored as numpy arrays, so you can use some nice numpy functions to make this math easier.

For finding the difference between the images, remember that, in code, an image is just an array of numbers. You can do arithmatic with the images to find how close they are to the same.

### converting to useful output

Back in the `predict` method, the output of `compare_images` passes through `_compute_prediction` and is saved in the dictionary `visual_diff`, which maps the keys associated with the template images to the calulated difference between that template image and the input image.

The final step for `TemplateMatcher` is to convert these differences into a scaled confidence value representing how well the input image matches each of the templates. This step is really just algebra: you need to make large numbers small and small numbers large, but you also want to scale your output so that a value near 1 always represents a high confidence.

### conclusion

That's all, this class now outputs how well an input image matches each of a given set of templates. One nice part about this approach is that no single step needs to be perfectly tuned: finding slightly too many keypoints initially is quickly corrected when you find matches, and incorrect matches are generally eliminated in the homography matrix transformation, so by the time you get to the numerical comparison of images, you are usually looking at either a reasonable match or two extremely different images. This system is therefore relatively robust, and has a low rate of false positives; however, it is suseptible to false negatives.

## Navigating
