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
