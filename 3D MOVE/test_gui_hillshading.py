from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from scipy.fftpack import diff
import pyautogui

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

screen_width, screen_height = pyautogui.size()          # 1920 x 1000
current_mouse_x, current_mouse_y = pyautogui.position()


print(current_mouse_x)
print(current_mouse_y)

print()
print(screen_height)
print(screen_width)


initial_x, initial_y = 10, 180

ax.view_init(initial_x, initial_y)      # So, this is basically x and y axes being defined.      x -> elevation  y -> azimuth
# So basically this will be used to move the graph

mouse_x_old, mouse_y_old = pyautogui.position()

display = True

while(display):
    mouse_y, mouse_x = pyautogui.position()
    print()
    print('y-co-ordinate: {}'.format(mouse_y))
    print('x-co-ordinate: {}'.format(mouse_x))

    diff_x, diff_y = mouse_x_old - mouse_x , mouse_y_old - mouse_y
    print()
    print(diff_x)
    print(diff_y)

    # instead of calculating difference, map/normalize the position values to reflect the values of the Azimuth and Elevation
    
    if diff_x > 180:
        diff_x = 180
    if diff_y > 180:
        diff_y = 180
    
    ax.view_init(initial_x + diff_x, initial_y + diff_y)

    ax.dist = 5            # for zooming and stuff....... Greater value would mean a smaller graph

    if mouse_y >= 1800:
        display = False
    if mouse_x >= 950:
        display = False

    ###################################################
    # Uncomment for full screen
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    ###################################################
    plt.draw()
    plt.pause(0.001)




plt.show()


# plt.close()
# plt.show()




# mng = plt.get_current_fig_manager()
# # mng.full_screen_toggle()
# mng.window.state('zoomed')
# plt.show()


