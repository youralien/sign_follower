# Sign Detector
This project, based on one by [Ryan Louie](https://github.com/youralien), Kai Levy, and Dakota Nelson, was provided as a scaffolded way to learn the basics of CV in ROS by the [Computational Robotics](https://sites.google.com/site/comprobo17/) class at Olin College of Engineering. In this project, images from a live camera on a Neato robot are filtered, cropped, fed through image recognition that uses SIFT to identify and match points between the live images and a set of templates. The templates cover left, right, and uturn signs, all of which can be recognized consistently in real time.

## System

### Detecting Signs in Scene Images
To make it easier on the SIFT processing, the Sign Detector first isolates yellow areas in the image that could be a sign, and crops the image around these areas. To do that, it converts the image into the HSV colorspace, filters out all but the yellow (the filter can be customized dynamically through the provided sliders), and converts to a black-and-white map. This map is subdivided into cells, and the number of white pixels in each section of the grid determines the sections included in the crop.

[filter_visual]: images/filter_screenshot.png "Scene before and after the filter, with a bounding box where the image will be cropped. You can also see the dynamic reconfigure sliders in the bottom left."

### Recognition

Recognizing the signs involves determining how well the cropped image if the sign matches the template image for each type of sign, and is done in the TemplateMatcher class. To do this, SIFT is used to find keypoints in the template image and in the input image, the images are then aligned with a brute force matcher, and compared. A value for how close each of the template images are to the current is fed back to the street sign recognizer, which displays the closest sign as the program is being run.

## Running
These instructions assume you have the Neato packages found at https://github.com/paulruvolo/comprobo17/tree/master/neato_robot.

To start the sign recognizer, run ROS, and if you've never used the sign_follower package before, do a catkin_make. To use the package with a live Neato, simply connect and run. To use the provided bags, make sure to also run the commands

```
rosparam set /use_sim_time true
```
```
roslaunch neato_node set_urdf.launch
```
```
rosrun image_transport republish compressed in:=/camera/image_raw _image_transport:=compressed raw out:=/camera/image_raw
```
and from the scripts directory:
```
rosbag play ../bags/uturn.bag -l
```
```
rosrun sign_follower street_sign_recognizer.py
```
(leftturn and rightturn bags are also available)
