import cv2
import os

imgpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../images/leftturn_scene.jpg")
img = cv2.imread(imgpath)

grid_cell_w = 64*0
grid_cell_h = 48*0

cv2.namedWindow("my_window")
# cv2.circle(img, (320, 240,), 10, 255, )
# NumPy array slicing!!
grid_cell = img[grid_cell_h:grid_cell_h + 48, grid_cell_w:grid_cell_w+64, ]

cv2.imshow("my_window", grid_cell)
cv2.waitKey(0);
