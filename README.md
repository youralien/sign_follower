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

__DELIVERABLE FOR THIS SECTION__: Visualize all these steps and save those results as a screenshot, screen capture, or saved image. And of course, write the code that produces these results.

#### Red-Green-Blue to Hue-Saturation-Value Images

There are different ways to represent the information in an image. A gray-scale iamge has `(n_rows, n_cols)`. An rgb image has shape `(n_rows, n_cols, 3)` since it has three channels: red, green, and blue.

Color images are also represented in different ways too.  Aside from the default RGB colorspace, there exists alot of others. We'll be focused on using [HSV/HSL](https://en.wikipedia.org/wiki/HSL_and_HSV): Hue, Saturation, and Value/Luminosity. Like RGB, a HSV image has three channels and is shape `(n_rows, n_cols, 3)`. The hue channel is well suited for color detection tasks, because we can filter by color on a single dimension of measurement, and it is a measure that is invariant to lighting conditions.  

[OpenCV provides methods to convert images from one color space to another.](http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor).

You will be writing all your image processing pipeline within the `process_image` callback function. Here is what the starter code looks like so far.

```python
    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        (...)

        # Creates a window and displays the image
        cv2.imshow('video_window', self.cv_image)
        cv2.waitKey(5)
```

A good first step would be convert `self.cv_image` into an HSV image and visualize it. Like any good roboticist, visualize everything to make sure it meets your expectations.

#### Filtering the image for only yellow

Since the set of signs we are recognizing are all yellow, by design, we can handtune a filter that will only select the certain shade of yellow in our image.

The starter code provides some helper methods

* What are some distinguishing visual features about the sign?  Is there similarities in color and/or geometry?
* Since we interested in generating a bounding box to be used in cropping out the sign from the original frame, what are different methods of generating candidate boxes?
* What defines a good bounding box crop?  It depends a lot on how robust the sign recognizer you have designed.

One method that could be fruitful would be dividing the image in a grid.  The object then is to decide which grid cells contain the region of interest.

![][grid]
[grid]: images/grid.png

Some suggested deliverables for this step
1. Use the provided test images `images/leftturn_scene.jpg` to develop a sign localization algorithm.
2. Drawing a way 

## Recognition

## Navigating
