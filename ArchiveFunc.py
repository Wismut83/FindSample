import matplotlib.pyplot as plt
import math

def circle():
    diameter = 10
    angle = 0
    smoothing = 360

    perimeter = math.pi*(diameter/2)*2
    radius = diameter/2

    alphas = [2*math.pi*rad/smoothing for rad in range(0,smoothing+1)]
    circul_x = []
    circul_y = []

    for alpha in alphas:
        x = radius * math.sin(alpha)
        y = radius * math.cos(alpha)
        circul_x.append(x)
        circul_y.append(y)



    plt.plot(circul_x, circul_y)
    plt.xlim(-1.1*radius,1.1*radius)
    plt.ylim(-1.1*radius,1.1*radius)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class detect_angle_skeleton():
    def __init__(self, img):
        self.img = img
        start_x, start_y = self.search_start_line(0)
        final_x, final_y = self.search_line(start_x, start_y)
        self.angle = math.degrees(math.atan((final_x - start_x) / (final_y - start_y)))

    def search_start_line(self, x):
        y = self.img.shape[1] // 2
        pixel = self.img[x][y][1]
        if pixel == 0:
            x += 1
            return self.search_start_line(x)
        if pixel >= 1:
            return x, y

    def search_line(self, x, y,  x_end = 0, y_end = 0):
        mask = [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]]  # по часовой
        for n in mask:
            x_next, y_next = x + n[0], y + n[1]
            if (x_next, y_next) == (x_end, y_end):
                return x_end, y_end
            elif self.img[x_next][y_next][1] > 0:
                x_end, y_end = x, y
                x_next, y_next = self.search_line(x_next, y_next, x_end, y_end)
                return x_next, y_next
            else:
                continue

def generate2class():
    from libraryNaN import images_dir
    from skimage.io import imread, imsave
    import os
    path_input = 'Photo/learn/2class/mask/'
    images = images_dir(path_input)
    for i, img  in enumerate(images):
        imsave(path_input+os.path.splitext(os.listdir(path_input)[i])[0]+'.png',imread(img)>200)

if __name__ == '__main__':
    generate2class()