#!/usr/bin/python
from logic import sift_flann_localizer
import os

os.chdir('/home/dhash/catkin_ws/src/sign_localizer/scripts')

dhash = sift_flann_localizer()
dhash.run(10)
