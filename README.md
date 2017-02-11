# Sign Follower

Here's a sample processing pipeline for doing robot sign following.

* __Localize__ the sign.  How does one draw a bounding box around the sign?
* __Recognize__ the sign's meaning. Is it telling me to turn right, left, or u-turn?  
* __Navigate__ based on the sign's message. 

## Localization

The Goal:

![][yellow_sign_detector]

#### Some guidance

* What are some distinguishing visual features about the sign?  Is there similarities in color and/or geometry?
* Since we interested in generating a bounding box to be used in cropping out the sign from the original frame, what are different methods of generating candidate boxes?
* What defines a good bounding box crop?  It depends a lot on how robust the sign recognizer you have designed.

One method that could be fruitful would be dividing the image in a grid.  The object then is to decide which grid cells contain the region of interest.

![][grid]

Some suggested deliverables for this step
1. Use the provided test images `images/leftturn_scene.jpg` to develop a sign localization algorithm.
2. Drawing a way 

## Recognition

## Navigating
[yellow_sign_detector]: images/yellow-sign-detector.gif
[grid]: images/grid.png
