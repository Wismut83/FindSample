import matplotlib.pyplot as plt
import math
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

