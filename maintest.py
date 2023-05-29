import numpy as np
from mesh_loader import mesh_loader
from BC_INIT_TEST import load_BC_init
from area_calc import area_normals_calc
from config import supersonic

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
prim, BC_left, BC_bot, BC_right, BC_up = load_BC_init(phi_rho, phi_u, phi_v, phi_p, imax, jmax, x_cell, y_cell)
A_vert, A_hori, normals_h, normals_v = area_normals_calc(imax, jmax, x, y)

def van_leer_flux():
    ''