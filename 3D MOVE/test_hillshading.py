from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

import numpy as np

# load and format data
demo_data = cbook.get_sample_data('jacksboro_fault_dem.npz', np_load=True)
z = demo_data['elevation']
nrows, ncols = z.shape

x = np.linspace(demo_data['xmin'], demo_data['xmax'], ncols)
y = np.linspace(demo_data['ymin'], demo_data['ymax'], nrows)
x, y = np.meshgrid(x, y)


region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

# set up a plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

# fig = plt.figure(figsize=(720, 980))

ls = LightSource(270, 45)

# To use a custom hillshading mode, over-ride the existing shading
# pass in rgb colors calculated from shade
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)

# count = 0
# while count < 2:
#     for angle in range(0, 360):
#         ax.view_init(30, angle)
#         plt.draw()
#         plt.pause(0.001)
#     count += 1

    # plt.show()


# plt.close()

ax.view_init(10, 180)      # So, this is basically x and y axes being defined.      x -> elevation  y -> azimuth
# So basically this will be used to move the graph


plt.show()
