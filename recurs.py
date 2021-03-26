import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
from libraryNaN import renumerate, images_dir, speed_test, detect_angle_measure, test_image
from skimage import measure
a = speed_test()

photo_number = 3
images = images_dir('Photo/test/mask/')
image_n, max_class = renumerate(imread(images[photo_number]), max=True)
image = renumerate(rotate(image_n, detect_angle_measure(image_n, max_class, image_n.shape[0]), mode='edge', order=0))

start, end = (test_image(image, max_class))

# label_img = measure.label(image < max_class)
# regions = measure.regionprops(label_img)

print(start, end)
fig, ax = plt.subplots()

for i in range(0, len(start)):
    ax.plot((0, len(image[0])), (start[i], start[i]), '-b')
for i in range(0, len(end)):
    ax.plot((0, len(image[0])), (end[i], end[i]), '-r')

# 50% от максиального по площади

a.stop()

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
