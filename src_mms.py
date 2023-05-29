from config import supersonic
import numpy as np
from mesh_loader import mesh_loader

L = 1

if supersonic == 1:
    phi_rho = np.array([1.0, 0.15, -0.1, 1.0, 0.5]) # phi0 phix phiy aphix aphiy
    phi_u   = np.array([800.0, 50.0, -30.0, 1.5, 0.6])
    phi_v   = np.array([800.0, -75.0, 40.0, 0.5, 2/3])
    phi_p   = np.array([1.0e5, 0.2e5, 0.5e5, 2, 1])
elif supersonic == 0:
    phi_rho = np.array([1.0, 0.15, -0.1, 1.0, 0.5]) # phi0 phix phiy aphix aphiy
    phi_u   = np.array([70.0, 5.0, -7.0, 1.5, 0.6])
    phi_v   = np.array([90.0, -15.0, 8.5, 0.5, 2/3])
    phi_p   = np.array([1.0e5, 0.2e5, 0.5e5, 2, 1])
else:
    print('E: Invalid value in supersonic flag, please check the config.py file')
    raise SystemExit(0)
    
imax, jmax, x, y, x_cell, y_cell = mesh_loader()

