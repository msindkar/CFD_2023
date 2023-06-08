from mesh_loader import mesh_loader
import matplotlib.pyplot as plt
import numpy as np
imax, jmax, x, y, x_cell, y_cell = mesh_loader()
plt.scatter(x, y, s=0.01)
plt.grid()
plt.show()