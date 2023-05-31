import numpy as np

from mesh_loader  import mesh_loader
from BC_INIT_TEST import load_BC_init
from area_calc    import area_normals_calc
from src_mms      import src_mms
from config       import flux_scheme

R = 8314        # J/(kmol*K)
m_air = 28.96   # 
R_air = R/m_air # Specific gas constant for air
nmax  = 10000   # max iterations
j = 0           # current iteration (REMOVE WHEN DONE)
gamma = 1.4

imax, jmax, x, y, x_cell, y_cell = mesh_loader()
prim1, T1, BC_left, BC_bot, BC_right, BC_up, BC_left_T, BC_bot_T, BC_right_T, BC_up_T = load_BC_init(R_air, imax, jmax, x_cell, y_cell)
A_vert, A_hori, normals_h, normals_v = area_normals_calc(imax, jmax, x, y)
src_array = src_mms(R_air, imax, jmax, x_cell, y_cell)

imaxc, jmaxc = imax - 1, jmax - 1

prim = np.zeros((imaxc + 4, jmaxc + 4, 4))
prim[2:imaxc + 2, 2:jmaxc + 2, :] = prim1[:, :, :]

T    = np.zeros((imaxc + 4, jmaxc + 4))
T[2:imaxc + 2, 2:jmaxc + 2] = T1[:, :]

del T1, prim1

alias_ci = list(np.arange(2, imaxc + 2))
alias_cj = list(np.arange(2, jmaxc + 2))

def extrapolate_to_ghost_cells():
    prim[alias_ci, 1, :] = 2*prim[alias_ci, 2, :] - prim[alias_ci, 3, :]
    prim[alias_ci, 0, :] = 2*prim[alias_ci, 1, :] - prim[alias_ci, 2, :]
    
    prim[alias_ci, jmaxc + 2, :]  = 2*prim[alias_ci, jmaxc + 1, :] - prim[alias_ci, jmaxc, :]
    prim[alias_ci, jmaxc + 3, :]  = 2*prim[alias_ci, jmaxc + 2, :] - prim[alias_ci, jmaxc + 1, :]
    
    prim[1, alias_cj, :] = 2*prim[2, alias_cj, :] - prim[3, alias_cj, :]
    prim[0, alias_cj, :] = 2*prim[1, alias_cj, :] - prim[2, alias_cj, :]
    
    prim[imaxc + 2, alias_cj, :] = 2*prim[imaxc + 1, alias_cj, :] - prim[imaxc, alias_cj, :]
    prim[imaxc + 3, alias_cj, :] = 2*prim[imaxc + 2, alias_cj, :] - prim[imaxc + 1, alias_cj, :]
    
    T[alias_ci, 1] = 2*T[alias_ci, 2] - T[alias_ci, 3]
    T[alias_ci, 0] = 2*T[alias_ci, 1] - T[alias_ci, 2]
    
    T[alias_ci, jmaxc + 2]  = 2*T[alias_ci, jmaxc + 1] - T[alias_ci, jmaxc]
    T[alias_ci, jmaxc + 3]  = 2*T[alias_ci, jmaxc + 2] - T[alias_ci, jmaxc + 1]
    
    T[1, alias_cj] = 2*T[2, alias_cj] - T[3, alias_cj]
    T[0, alias_cj] = 2*T[1, alias_cj] - T[2, alias_cj]
    
    T[imaxc + 2, alias_cj] = 2*T[imaxc + 1, alias_cj] - T[imaxc, alias_cj]
    T[imaxc + 3, alias_cj] = 2*T[imaxc + 2, alias_cj] - T[imaxc + 1, alias_cj]
extrapolate_to_ghost_cells()

a = np.zeros((imaxc + 4, jmaxc + 4))
def a_calc():
    a[:, :] = (gamma*R_air*T[:, :])**(0.5)
    a[a[:, :] == 0] = 1
a_calc()

ht   = np.zeros((imaxc + 4, jmaxc + 4))
vel2 = np.zeros((imaxc + 4, jmaxc + 4))
def ht_calc():
    vel2[:, :] = prim[:, :, 1]**2 + prim[:, :, 2]**2
    ht  [:, :] = ((gamma*R_air)/(gamma - 1))*T[:, :] + (vel2[:, :]**2)/2
ht_calc()

def M_calc(): # Think about this one
    ''

def mms_boundary_conditions():
    prim[alias_ci, 2] = BC_bot[:]
    prim[2, alias_cj] = BC_left[:]
    prim[alias_ci, jmaxc + 1] = BC_up[:]
    prim[imaxc + 1, alias_cj] = BC_right[:]
    
    T[alias_ci, 2] = BC_bot_T[:]
    T[2, alias_cj] = BC_left_T[:]
    T[alias_ci, jmaxc + 1] = BC_up_T[:]
    T[imaxc + 1, alias_cj] = BC_right_T[:]

def van_leer_flux(j):
    # for i in range(imax + 1):
    #     M_L = primitive_variables[1, i]/a[0, i]
    #     M_R = primitive_variables[1, i + 1]/a[0, i + 1]
    #     M_plus = (1/4)*(M_L + 1)**2
    #     M_minus = -(1/4)*(M_R - 1)**2
    #     beta_L = -max(0, (1 - int(M_L)))
    #     beta_R = -max(0, (1 - int(M_R)))
    #     alpha_plus = (1/2)*(1 + np.sign(M_L))
    #     alpha_minus = (1/2)*(1 - np.sign(M_R))
    #     c_plus = alpha_plus*(1 + beta_L)*M_L - beta_L*M_plus
    #     c_minus = alpha_minus*(1 + beta_R)*M_R - beta_R*M_minus
        
    #     F_C_p = primitive_variables[0, i]*a[0, i]*c_plus*np.array([1, primitive_variables[1, i], ht[0, i]])
    #     F_C_m = primitive_variables[0, i + 1]*a[0, i + 1]*c_minus*np.array([1, primitive_variables[1, i + 1], ht[0, i + 1]])
        
    #     P_2bar_plus = M_plus*(- M_L + 2)
    #     P_2bar_minus = M_minus*(- M_R - 2)
    #     D_plus = alpha_plus*(1 + beta_L) - beta_L*P_2bar_plus
    #     D_minus = alpha_minus*(1 + beta_R) - beta_R*P_2bar_minus
        
    #     F_P_p = np.array([0, D_plus*primitive_variables[2, i], 0])
    #     F_P_m = np.array([0, D_minus*primitive_variables[2, i + 1], 0])
        
    #     F[:, i] = F_C_p + F_P_p + F_C_m + F_P_m
    
    ''