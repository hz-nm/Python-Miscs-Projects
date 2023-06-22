from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some test data
X, Y, Z = axes3d.get_test_data(0.05)

# plot a basic wireframe
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

plt.show()