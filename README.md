# Sign Follower

By the end of the Traffic Sign Follower project, you will have a robot that will look like this! 

**Video Demo:** [NEATO ROBOT OBEYS TRAFFIC SIGNS](https://youtu.be/poReVhj1lSA)

## System
![alt text][system-overview]

[system-overview]: images/vision-nav-system-overview.png "Three stages of the vision and navigation system: 1) waypoint navigation 2) sign recognition, and 3) sign obeyance via changing the next waypoint"

Navigation and mapping is handled by the built-in ROS package ```neato_2dnav``` .  Mapping the location of the robot in the environment was handled by [```gmapping```](http://wiki.ros.org/gmapping), a package that provides laser-based SLAM (simultaneous localization and mapping).  Navigation was handled by the [```move_base```](http://wiki.ros.org/move_base) package;   our program published waypoints to the ```/move_base_simple/goal``` topic while the internals of path planning and obstacle avoidance were abstracted away.

You will put your comprobo-chops to the test by developing a sign detection node which publishes to a topic, ```/predicted_sign```, once it is confident about recognizing the traffic sign in front of it. 

### Running and Developing the street_sign_recognizer node

We've provided some rosbags that will get you going.

[uturn.bag](https://drive.google.com/open?id=0B85lERk460TUYjFGLVg1RXRWams)
[rightturn.bag](https://drive.google.com/open?id=0B85lERk460TUN3ZmUk15dmtPTFk)
[leftturn.bag](https://drive.google.com/open?id=0B85lERk460TUTkdTQW5yQ0FwSEE)

Start by replaying these rosbags on loop:

```
rosbag play uturn.bag -l
```

They have `/camera/image_raw/compressed` channel recorded. In order to republish the compressed image as a raw image, 

```
rosrun image_transport republish compressed in:=/camera/image_raw _image_transport:=compressed raw out:=/camera/image_raw
```

To run the node,

```
rosrun sign_follower street_sign_recognizer.py
```

If you ran on the steps above correctly, a video window should appear visualizing the Neato moving towards a traffic sign. 

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

        pt1, pt2 = self.sign_bounding_box()
        
        # some other code to crop the bounding box
        # and recognize the traffic sign
        (...)

        # Creates a window and displays the image
        cv2.imshow('video_window', self.cv_image)
        cv2.waitKey(5)
```

The goal of localizing the signs in the scene is to determine `pt1 = (x1,y1)` and `pt2 = (x2,y2)` points that define the upper lefthand corner and lower righthand corner of a bounding box around the sign. You can do most of your work in the instance method `sign_bounding_box` if you define your working variables as instance variables, i.e. `self.cv_image`.

Whether you follow along with the suggested steps for creating a sign recognizer or have ideas of your own, revisit these questions often when designing your image processing pipeline:

* What are some distinguishing visual features about the sign?  Is there similarities in color and/or geometry?
* Since we are interested in generating a bounding box to be used in cropping out the sign from the original frame, what are different methods of generating candidate boxes?
* What defines a good bounding box crop?  It depends a lot on how robust the sign recognizer you have designed.

Finally, if you think that working with individual images, outside of the `StreetSignRecognizer` class would be helpful -- I often like to prototype the algorithms I am developing in a jupyter notebook -- feel free to use some of the image frames in the `images/` folder.  In addition, you can save your own images from the video feed by setting the flag

```python
self.use_saver = True
```

The images currently save in your `/tmp/` directory. See the `process_image` callback for more details.

#### Red-Green-Blue to Hue-Saturation-Value Images

There are different ways to represent the information in an image. A gray-scale iamge has `(n_rows, n_cols)`. An rgb image has shape `(n_rows, n_cols, 3)` since it has three channels: red, green, and blue.

Color images are also represented in different ways too.  Aside from the default RGB colorspace, there exists alot of others. We'll be focused on using [HSV/HSL](https://en.wikipedia.org/wiki/HSL_and_HSV): Hue, Saturation, and Value/Luminosity. Like RGB, a HSV image has three channels and is shape `(n_rows, n_cols, 3)`. The hue channel is well suited for color detection tasks, because we can filter by color on a single dimension of measurement, and it is a measure that is invariant to lighting conditions.  

[OpenCV provides methods to convert images from one color space to another](http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor).

A good first step would be convert `self.cv_image` into an HSV image and visualize it. Like any good roboticist, visualize everything to make sure it meets your expectations.

#### Filtering the image for only yellow

Since the set of signs we are recognizing are all yellow, by design, we can handtune a filter that will only select the certain shade of yellow in our image.

Look for the lines where

```python
        self.use_slider = False
        self.use_mouse_hover = False
```

Setting these flags to True will turn on GUI elements associated with the OpenCV window.

If you hover over a certain part of the image, it will tell you what R, G, B value you are hovering over. The details are in the `process_mouse_event` method.  Once you have created an HSV image, you can also edit this function to also display the Hue, Saturation, and Value numbers.

The sliders will help set the hsv lower and upper bound limits (`self.hsv_lb` and `self.hsv_ub`), which you can then use as limits for filtering certain parts of the HSV spectrum. Check out the OpenCV [inRange method](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html) for more details on how to threshold an image for a range of a particular color.

By the end of this step, you should have a binary image mask where all the pixels that are white represent the color range that was specified in the thresholding operation.

#### Generating a bounding box 

You can develop an algorithm that operates on the binary image mask that you developed in the step above.

One method that could be fruitful would be dividing the image in a grid.  The object then is to decide which grid cells contain the region of interest.

You might want to write a method that divides the image into a binary grid of `grid_size=(M,N)`; if tile in the grid contains a large enough percentage of white pixels, the tile will be turned on.

Then you can write another function that takes this binary grid and determines the bounding box that will include all the grid cells that were turned on.

![][grid]
[grid]: images/grid.png

OpenCV has a method called `boundingRect` which seems promising too.  I found a nice example, albeit in C++, that [finds a bounding rectangle using a threshold mask](http://answers.opencv.org/question/4183/what-is-the-best-way-to-find-bounding-box-for-binary-mask/) like you have.  It seems like it will depend on your thresholding operation to be pretty clean (i.e. no spurious white points, the only object that should be unmasked is the sign of interest). 

![][boundingRectStars]
[boundingRectStars]: images/boundingRectStars.png

The goal is to produce `pt1 = (x1, y1)` and `pt2 = (x2, y2)` that define the upperleft and lower right corners of the bounding box.  Remember that the quality of the bounding box you need to produce will depend on how robust later steps of your computer vision pipeline are. 

## Recognition

Recognizing the signs involves determining how well the cropped image if the sign matches the template image for each type of sign. To do this, we will find keypoints in the template image and in the input image, then see how well we can align the keypoints, and finally see how similar the aligned images are.

### finding keypoints

We are finding keypoints using open cv's implementation of the [SIFT algorithm](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html), then filtering the keypoints ourselves to find the points that match between the input image and each template.

For the template images, the keypoints are calculated once, in `TemplateMatcher`'s init method:
``` python
self.kps[k], self.descs[k] = self.sift.detectAndCompute(self.signs[k],None)
```
The first TODO in the `predict` method is to find the keypoints in the input image as the first step to matching it with at template.

### aligning keypoints

In `_compute_prediction`, the first step to aligning the images is to find which keypoints from the two images match. The simplest method for that is to say keypoints whcih are close to eachother match. (note, later steps will correct for "matched" keypionts from mismatched images)

Based on the matched keypoints, the following lines find how to transform the input image so the keypoints align with the template images keypoints (using a homography matrix), then transorms the input image using that matrix, so it should align with the template.
``` python
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
img_T = cv2.warpPerspective(img, M, self.signs[k].shape[::-1])
```
### comparing images

At this point, we have two images which, if they are of the same sign, should be aligned with each other. If they are of different signs, the matched keypoints were likely not well aligned, and the homography matrix probably skewed the image into an unrecognizable blob, but the computer can't tell what is a reasonable image and what is an unrecognizable blob, so now we have to determine how similar the two images are.

The `compare_images` function at the top of the file is used to find how similar two images are to each other. This one is left up to you, but here are a few hints:

First, there is one thing we have yet to account for while comparing images: lighting. If you have tried to do blob detection and then tried again when the sun went down, you know that lighting wreaks havoc on computer vision. Since we are using grayscale images, and we have cropped them so both images are of the same thing, we can eliminate most of the effects of lighting by normalizing the image. This should be implemented in the `normalize` function and used in `compare_images`

For finding the difference between the images, remember that, in code, an image is just an array of numbers. You can do arithmatic with the images to find how close they are to the same.

### converting to useful output

Back in the `predict` method, the output of `compare_images` passes through `_compute_prediction` and is saved in the dictionary `visual_diff`, which maps the keys associated with the template images to the calulated difference between that template image and the input image.

The final step for `TemplateMatcher` is to convert these differences into a scaled confidence value representing how well the input image matches each of the templates. This step is really just algebra: you need to make large numbers small and small numbers large, but you also want to scale your output so that a value near 1 always represents a high confidence.

### conclusion

That's all, this class now outputs how well an input image matches each of a given set of templates. One nice part about this approach is that no single step needs to be perfectly tuned: finding slightly too many keypoints initially is quickly corrected when you find matches, and incorrect matches are generally eliminated in the homography matrix transformation, so by the time you get to the numerical comparison of images, you are usually looking at either a reasonable match or two extremely different images. This system is therefore relatively robust, and has a low rate of false positives; however, it is suseptible to false negatives.

## Navigating
