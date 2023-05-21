import numpy as np
from mesh_loader import mesh_loader
from BC_INIT_TEST import load_BC_init

imax, jmax, x, y = mesh_loader()
prim, left, bot, right, up = load_BC_init(imax, jmax, x, y)