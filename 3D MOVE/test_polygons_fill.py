from matplotlib import projections
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.misc import face
from scipy.stats import poisson
import numpy as np

# Fix random state for reproducibility
np.random.seed(19680801)

def polygon_under_graph(x, y):
    # construct a vertex list which defines the polygon filling
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
    # assumes x is in ascending order.


ax = plt.figure().add_subplot(projection='3d')

x = np.linspace(0., 10., 31)
# print(x)

lambdas = range(1, 10)

# verts[i] is a list of (x, y) pairs defining a polygon i.
verts = [polygon_under_graph(x, poisson.pmf(l, x)) for l in lambdas]        # pmf -> Probability Mass Function = pmf(k, mu)
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

poly = PolyCollection(verts, facecolors= facecolors, alpha=0.7)

ax.add_collection3d(poly, zs=lambdas, zdir='y')

ax.set(xlim=(0, 10), ylim=(1, 9), zlim=(0, 0.35), xlabel='x', ylabel=r'$\lambda$', zlabel='probability')    # r'$\lambda$' --> gets the lambda logo

plt.show()

