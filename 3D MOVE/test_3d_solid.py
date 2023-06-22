# Example from matplotlib's documentation

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

# let's make some random data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 10 * np.outer(np.cos(u), np.sin(v))     # returns the outer product of two vectors.
y = 10 * np.outer(np.sin(u), np.sin(v))     # outer product -> element by element mupltiplication
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# let's now plot the surface
ax.plot_surface(x, y, z)

plt.show()