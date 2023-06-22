from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np

n_radii = 8
n_angles = 36

# Make radii and angle arrays (also called spaces) Here radius r=0 is omitted to eliminate duplication
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]     # np.newaxis -> used to increase the dimension of this array by one more dimension
                                                                                # so a 1D array will become a 2D array.

# convert polar co-ordinates to cartesian
# (0, 0) will be manually added so that there is no duplication.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# print(radii)
# print(angles)
# print(x)
# print(y)

# Compute Z corresponding to a pringle's surface.
z = np.sin(-x*y)

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)           # triangular surface. # Antialiasing = True prints much smoother graphs
                                                                                          # turn antialiasing off when data is large to get quicker results.

plt.show()
