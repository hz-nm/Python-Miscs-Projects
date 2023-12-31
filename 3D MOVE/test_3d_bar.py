import numpy as np
import matplotlib.pyplot as plt

# setup the figure and axes
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# let's generate some fake data
x = np.arange(4)
y = np.arange(5)

xx, yy = np.meshgrid(x, y)

_x, _y = xx.ravel(), yy.ravel()

top = _x + _y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(_x, _y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

ax2.bar3d(_x, _y, bottom, width, depth, top, shade=False)
ax2.set_title('Non-Shaded')

plt.show()
