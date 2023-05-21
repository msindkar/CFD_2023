# BC/INIT CALC
from mesh_loader import mesh_loader
from config import supersonic
import numpy as np

L = 1

imax, jmax, x, y = mesh_loader()

prim = np.zeros((imax, jmax, 4)) # rho u v p

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
    print('Invalid value in supersonic flag, please check the config.py file')
    exit()

prim[:, :, 0] = phi_rho[0] + phi_rho[1]*np.sin(phi_rho[3]*np.pi*x/L) + phi_rho[2]*np.cos(phi_rho[4]*np.pi*y/L)
prim[:, :, 1] = phi_u[0]   + phi_u[1]*np.sin(phi_u[3]*np.pi*x/L)     + phi_u[2]*np.cos(phi_u[4]*np.pi*y/L)
prim[:, :, 2] = phi_v[0]   + phi_v[1]*np.cos(phi_v[3]*np.pi*x/L)     + phi_v[2]*np.sin(phi_v[4]*np.pi*y/L)
prim[:, :, 3] = phi_p[0]   + phi_p[1]*np.cos(phi_p[3]*np.pi*x/L)     + phi_p[2]*np.sin(phi_p[4]*np.pi*y/L)