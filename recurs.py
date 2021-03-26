import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
from libraryNaN import renumerate, images_dir, speed_test, detect_angle_measure

a = speed_test()

photo_number = 3
images = images_dir('Photo/test/mask/')
image_n, max_class = renumerate(imread(images[photo_number]), max=True)
image = renumerate(rotate(image_n, detect_angle_measure(image_n, max_class, image_n.shape[0]), mode='edge', order=0))

def test_mask_line(line, white, index=0, part=0.05, test_result=0):
    max_black = len(line) * part
    for i in range(0, len(line)):
        if test_result>max_black:
            return True
        if line[index] != white:
            test_result += 1
        index += 1
    return False

def test_image(image, max_class):
    last, start, end = False, [],[]
    for i in range(0,len(image),1):
        line_bool = test_mask_line(image[i], max_class)
        if line_bool is True and last is False:
            start.append(i)
        elif line_bool is False and last is True:
            end.append(i)
        last = line_bool
    if len(start) != len(end):
        raise ValueError("Списки 'start' и 'end' разной длины")
    return start, end


start, end = (test_image(image, max_class))

print(start,end)
fig, ax = plt.subplots()
for i in range(0,len(start)):
    ax.plot((0,len(image[0])),(start[i],start[i]), '-b')
for i in range(0,len(end)):
    ax.plot((0,len(image[0])),(end[i],end[i]), '-r')

# 50% от максиального по площади

a.stop()


ax.imshow(image, cmap=plt.cm.gray)
plt.show()