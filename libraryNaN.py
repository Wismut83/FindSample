import math
import numpy as np
import time

def rgb2gray(rgb, r=0.3, g=0.4, b=0.3):
    try:
        r_k, g_k, b_k = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = r_k * r + g_k * g + b_k * b
        return gray
    except:
        return rgb

def images_dir(path, not_prefix=None, suffix=None):
    import os
    if not_prefix == None and suffix == None:
        return sorted([os.path.join(path, name) for name in os.listdir(path)])
    elif not_prefix != None and suffix != None:
        return sorted([os.path.join(path, name) for name in os.listdir(path) if name.endswith(suffix) and not name.startswith(not_prefix)])
    elif not_prefix != None and suffix == None:
        return sorted([os.path.join(path, name) for name in os.listdir(path) if not name.startswith(not_prefix)])
    elif not_prefix == None and suffix != None:
        return sorted([os.path.join(path, name) for name in os.listdir(path) if name.endswith(suffix)])

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    elif n == 1:
        return([start])
    else:
        return([])

def weighted_avarage_angle(regions):
    w, wx, minus = 0, 0, 0
    for props in regions:
        minus+=props.orientation
        wx+=props.area*abs(math.tan(props.orientation))
        w+=props.area
    angle = 90 - math.degrees(math.atan(wx/w))
    print('angle: ', angle)
    if minus>=0:
        return angle
    else:
        return -angle

def detect_angle_measure(img, max_class, max_area):
    from skimage import measure
    label_img = measure.label(img < max_class)
    regions = measure.regionprops(label_img)
    try_regions = [x for x in regions if x.area > max_area]
    # for props in try_regions:
    #     print(math.degrees(props.orientation))
    return weighted_avarage_angle(try_regions)

def renumerate(image, max = False):
    image = rgb2gray(image)
    classes = np.unique(image)
    dummy = np.zeros_like(image)
    for idx, value in enumerate(classes):
        mask = np.where(image == value)
        dummy[mask] = idx
    max_class = idx
    if max is True:
        return dummy, max_class
    if max is False:
        return dummy

class speed_test():
    def __init__(self):
        self.start = time.time()
    def stop(self):
        print("--- %s seconds ---" % (time.time() - self.start))