import math
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import data, filters, measure, morphology
from skimage.io import imread, imsave
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from libraryNaN import renumerate, images_dir, speed_test, detect_angle_measure

a = speed_test()


def draw_lines(regions):
    for props in regions:

        y0, x0 = props.centroid
        orientation = props.orientation
        or_i = -1
        if orientation < 0:
            or_i = 1
        miny, minx, maxy, maxx = props.bbox
        #
        # print(miny, minx, maxy, maxx)
        # print(props.minor_axis_length, props.major_axis_length, orientation)

        kx1, kx2 = x0 - minx, maxx - x0
        x1 = minx
        y1 = y0 + kx1 * math.cos(orientation) * or_i
        x2 = maxx
        y2 = y0 - kx2 * math.cos(orientation) * or_i

        ky1, ky2 = y0 - miny, maxy - y0
        x3 = x0 - ky1 * math.cos(orientation) * or_i
        y3 = miny
        x4 = x0 + ky2 * math.cos(orientation) * or_i
        y4 = maxy

        ax.plot((x1, x2), (y1, y2), '-r', linewidth=2.5)
        ax.plot((x3, x4), (y3, y4), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        bx = (minx, maxx, maxx, minx, minx)
        by = (miny, miny, maxy, maxy, miny)
        ax.plot(bx, by, '-b', linewidth=2.5)


photo_number = 28

images = images_dir('Photo/test/mask/')
print(images[photo_number])
image, max_class = renumerate(imread(images[photo_number]), max=True)
print(max_class)
rot_image = renumerate(rotate(image, detect_angle_measure(image, max_class, image.shape[0]), mode='edge', order=0))

# img_lines = []
# for i in range(0, len(rot_image), 3):
#     print(rot_image[i])


label_img = measure.label(rot_image<max_class)
regions = regionprops(label_img)
rot_regions = [x for x in regions if x.area>image.shape[0]]


fig, ax = plt.subplots()
ax.imshow(rot_image, cmap=plt.cm.gray)
draw_lines(rot_regions)


a.stop()
plt.show()
