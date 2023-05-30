import numpy as np
from mesh_loader import mesh_loader
from BC_INIT_TEST import load_BC_init
from area_calc import area_normals_calc
from src_mms import src_mms

R = 8314    # J/(kmol*K)
m_air = 28.96 # 
R_air = R/m_air # Specific gas constant for air

imax, jmax, x, y, x_cell, y_cell = mesh_loader()
prim, T, left, BC_bot, BC_right, BC_up, BC_left_T, BC_bot_T, BC_right_T, BC_up_T = load_BC_init(R_air, imax, jmax, x_cell, y_cell)
A_vert, A_hori, normals_h, normals_v = area_normals_calc(imax, jmax, x, y)
src_array = src_mms(R_air, imax, jmax, x_cell, y_cell)

def van_leer_flux():
    ''