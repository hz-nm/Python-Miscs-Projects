# environment --> conda activate telecon

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

def randrange(n, vmin, vmax):
    # function to make a random array
    # having shape (n, ) with values distributed between vmin and vmax
    
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

n = 100

# for each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh]
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)

    ax.scatter(xs, ys, zs, marker = m)

ax.set_xlabel('X-label')
ax.set_ylabel('Y-label')
ax.set_zlabel('Z-label')

plt.show()