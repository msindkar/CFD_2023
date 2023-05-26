import numpy as np
from mesh_loader import mesh_loader
from BC_INIT_TEST import load_BC_init
from area_calc import area_normals_calc

imax, jmax, x, y, x_cell, y_cell = mesh_loader()
prim, left, bot, right, up = load_BC_init(imax, jmax, x, y)
A_vert, A_hori, normals_h, normals_v = area_normals_calc(imax, jmax, x, y)