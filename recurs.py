import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate, resize
import libraryNaN as NaN
from skimage import measure


a = NaN.speed_test()

photo_number = 28
# Загружаем пути фото и маски
masks = NaN.images_dir('Photo/test/mask/')
images = NaN.images_dir('Photo/test/img/')

# Открываем маску и получаем количесво классов на ней
mask_n, max_class = NaN.renumerate(imread(masks[photo_number]), max=True, inver=True)

# Вычисляем угол поворота изображения
angle = NaN.detect_angle_measure(mask_n, mask_n.shape[0], max_class)

# Поворачиваем маску и фото
mask = NaN.renumerate(rotate(mask_n, angle, mode='edge', order=0))
image = rotate(imread(images[photo_number]),angle,mode='edge', order=0)

# Вычисляем максимальный и минимальный Y
y_min, y_max = NaN.minmax_y(mask, mask_n.shape[0], max_class)

# Вычисляем оси и ширину
axis = (NaN.image_axis(mask, max_class))

# Разбиваем картинуц на керны
kerns=[]
sqrt_list = []
for i in axis:
    kerns.append(image[i[1]:i[2], y_min:y_max])

# Разбиваем керны на квадраты
for k in kerns:
    overlap = 1.3
    max_x, max_y, n_colors = k.shape
    y_start, y_end = 0, max_x
    while y_end<max_y:
        sqrt_list.append(k[0:max_x,y_start:y_end])
        y_start = y_end-round(max_x*0.3)
        y_end = y_start+max_x
    sqrt_list.append(k[0:max_x,max_y-max_x:max_y])

for i, sq in enumerate(sqrt_list):
    file = resize(sq, (64,64), order=0, clip=True, anti_aliasing=False)
    NaN.save_list_img('Photo/test/img/sq/',str(i),file)


    # n_sqrt = math.ceil(y/x*overlap)
    # print(x, round(y/n_sqrt/2))

    # print(kern.shape[1]/kern.shape[0])
    # print(round((n_sqrt-kern.shape[1]/kern.shape[0])*kern.shape[0]/n_sqrt))
#
#     print('======')

    # kerns.append([])
    # for i in range(0,math.ceil(kern.shape[1]/kern.shape[0])):
    #     kerns[-1].append()


# # Рисование кернов
# fig, ax = plt.subplots(len(kerns))
# for i in range(0,len(kerns)):
#     ax[i].imshow(kerns[i],cmap=plt.cm.gray)
#
a.stop()
# plt.show()
