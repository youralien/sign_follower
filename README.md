# Sign Detector
This project, based on one by [Ryan Louie](https://github.com/youralien), Kai Levy, and Dakota Nelson, was provided as a scaffolded way to learn the basics of CV in ROS by the [Computational Robotics](https://sites.google.com/site/comprobo17/) class at Olin College of Engineering. In this project, images from a live camera on a Neato robot are filtered, cropped, fed through image recognition that uses SIFT to identify and match points between the live images and a set of templates. The templates cover left, right, and uturn signs, all of which can be recognized consistently in real time.

## System

### Detecting Signs in Scene Images
To make it easier on the SIFT processing, the Sign Detector first isolates yellow areas in the image that could be a sign, and crops the image around these areas. To do that, it converts the image into the HSV colorspace, filters out all but the yellow (the filter can be customized dynamically through the provided sliders), and converts to a black-and-white map. This map is subdivided into cells, and the number of white pixels in each section of the grid determines the sections included in the crop.

![][yellow_sign_detector]
[yellow_sign_detector]: images/yellow-sign-detector.gif "Bounding box generated around the yellow parts of the image.  The video is converted to HSV colorspace, an inRange operation is performed to filter out any non yellow objects, and finally a bounding box is generated."

## Recognition

Recognizing the signs involves determining how well the cropped image if the sign matches the template image for each type of sign. To do this, we will find keypoints in the template image and in the input image, then see how well we can align the keypoints, and finally see how similar the aligned images are.

### Running

To start the sign recognizer, run ROS, and if you've never used the sign_follower package before, do a catkin make. To use the package with a live Neato, simply connect and run. To use the provided bags, make sure to also run the commands
```
rosparam set /use_sim_time true
```
```
roslaunch neato_node set_urdf.launch
```
```
rosrun image_transport republish compressed in:=/camera/image_raw _image_transport:=compressed raw out:=/camera/image_raw
```
and from the pacakge directory:
```
rosbag play bags/uturn.bag -l
```
