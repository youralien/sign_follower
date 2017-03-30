import numpy as np

def of_interest(arr, start_x, start_y, box_width, box_height):
    count = 0
    #print(start_x, start_y)
    for i in range(start_y, start_y + box_height):
        for j in range(start_x, start_x + box_height):
            #print self.binary_image[i, j]
            if arr[i, j] > 0:
                count += 1
    print(count)
    if float(count)/(box_width*box_height) > 0.15:
        return True
    else:
        return False

arr = np.array([[1, 0], [1, 1]])

print(of_interest(arr, 0, 0, 2, 2))
