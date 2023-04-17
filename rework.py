# AOE6145: Computational Fluid Dynamics, 2023
# Homework 4

import numpy as np
from hw_1_func import supersonic_nozzle_exact_solution

p0 = 3E5 # Stagnation pressure, Pa
t0 = 600 # Stagnation temperature, K
rho0 = 1.7416406062063983 # Stagnation density, kg/m3 (calculated, check maybe?)
gamma = 1.4
R = 8314    # J/(kmol*K)
m_air = 28.96 # 
R_air = R/m_air # Specific gas constant for air
pback = 1.2E5

# const1 = (gamma - 1)/2
# const2 = gamma/(gamma - 1)
# const3 = 2/(gamma - 1)

nmax = 200000 # no. of iterations
cfl = 0.01
kappa2 = 0.5
kappa4 = 1/32
# ---------- Set geometry ----------

print('Enter number of cells (even number):')
imax = int(input())

dx = 2/imax
x_cell = np.arange(-1 + 1/imax, 1 + 1/imax, 2/imax)      # x-position of cell centers
x_intf = np.arange(-1, 1 + 1/imax, 2/imax)               # x-position of cell interfaces
x_intf[np.argmin(np.abs(x_intf))] = 0                    # set x = 0 at throat
A_cell = 0.2 + 0.4*(1 + np.sin(np.pi*(x_cell - 0.5)))    # area at cell centers
A_intf = 0.2 + 0.4*(1 + np.sin(np.pi*(x_intf - 0.5)))    # area at cell interfaces
dA_dx  = 0.4*np.pi*np.cos(np.pi*x_cell - np.pi*0.5)

V = A_cell*dx # cell volume based on areas a cell centers

# ---------- Check case ----------

print('1 for isentropic case, anything else for shock case')
if int(input()) == 1:
    shock_flag = 0
    exact_solution = supersonic_nozzle_exact_solution(p0, t0, imax)
else:
    shock_flag = 1
    #pback = 1.2E5 # Back pressure for shock case, Pa
    
# ---------- Set inital conditions ----------

primitive_variables = np.zeros((3, imax + 2))    # Array of cell centers, even number, 2 ghost cells, rho u P
cell_alias = list(range(1, imax + 1))     # Alias range for non-ghost cells easier indexing
#intf_alias = list(range(1, imax + 2))     # Alias range for non-ghost interfaces easier indexing
M = np.zeros((1, imax + 2))
T = np.zeros((1, imax + 2))
a = np.zeros((1, imax + 2))
ht = np.zeros((1, imax + 2))
et = np.zeros((1, imax + 2))

M[0, cell_alias] = x_cell*1.4 + 1.6

psi = 1 + ((gamma - 1)/2)*M[0, cell_alias]**2
T[0, cell_alias] = t0/psi
primitive_variables[2, cell_alias] = p0/(psi**(gamma/(gamma - 1)))
primitive_variables[0, cell_alias] = primitive_variables[2, cell_alias]/(R_air*T[:, cell_alias])
a[0, cell_alias] = (gamma*R_air*T[0, cell_alias])**(0.5)
primitive_variables[1, cell_alias] = M[0, cell_alias]*a[0, cell_alias]

ht[0, cell_alias] = ((gamma*R_air)/(gamma - 1))*T[0, cell_alias] + (primitive_variables[1, cell_alias]**2)/2
et[0, cell_alias] = ht[0, cell_alias] - (primitive_variables[2, cell_alias]/primitive_variables[0, cell_alias])

# ---------- ---------- ----------

def upwind_boundary_conditions():
    M[0, 0] = 2*M[0, 1] - M[0, 2]
    
    psi_bc_0 = 1 + ((gamma - 1)/2)*M[0, 0]**2
    T[0, 0] = t0/psi_bc_0
    primitive_variables[2, 0] = p0/(psi_bc_0**(gamma/(gamma - 1)))
    primitive_variables[0, 0] = primitive_variables[2, 0]/(R_air*T[:, 0])
    a[0, 0] = (gamma*R_air*T[0, 0])**(0.5)
    primitive_variables[1, 0] = M[0, 0]*a[0, 0]\
    
    ht[0, 0] = ((gamma*R_air)/(gamma - 1))*T[0, 0] + (primitive_variables[1, 0]**2)/2
    et[0, 0] = ht[0, 0] - (primitive_variables[2, 0]/primitive_variables[0, 0])
    
    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]
    
    psi_bc_1 = 1 + ((gamma - 1)/2)*M[0, imax + 1]**2
    T[0, imax + 1] = t0/psi_bc_1
    primitive_variables[2, imax + 1] = p0/(psi_bc_1**(gamma/(gamma - 1)))
    primitive_variables[0, imax + 1] = primitive_variables[2, imax + 1]/(R_air*T[:, imax + 1])
    a[0, imax + 1] = (gamma*R_air*T[0, imax + 1])**(0.5)
    primitive_variables[1, imax + 1] = M[0, imax + 1]*a[0, imax + 1]
    
    ht[0, imax + 1] = ((gamma*R_air)/(gamma - 1))*T[0, imax + 1] + (primitive_variables[1, imax + 1]**2)/2
    et[0, imax + 1] = ht[0, imax + 1] - (primitive_variables[2, imax + 1]/primitive_variables[0, imax + 1])

upwind_boundary_conditions()

F = np.zeros((3, imax + 1))

def van_leer_1st_order_flux():
    for i in range(imax + 1):
        M_L = primitive_variables[1, i]/a[0, i]
        M_R = primitive_variables[1, i + 1]/a[0, i + 1]
        M_plus = (1/4)*(M_L + 1)**2
        M_minus = -(1/4)*(M_R - 1)**2
        beta_L = -max(0, (1 - int(M_L)))
        beta_R = -max(0, (1 - int(M_R)))
        alpha_plus = (1/2)*(1 + np.sign(M_L))
        alpha_minus = (1/2)*(1 - np.sign(M_R))
        c_plus = alpha_plus*(1 + beta_L)*M_L - beta_L*M_plus
        c_minus = alpha_minus*(1 + beta_R)*M_R - beta_R*M_minus
        
        F_C_p = primitive_variables[0, i]*a[0, i]*c_plus*np.array([1, primitive_variables[1, i], ht[0, i]])
        F_C_m = primitive_variables[0, i + 1]*a[0, i + 1]*c_minus*np.array([1, primitive_variables[1, i + 1], ht[0, i + 1]])
        
        P_2bar_plus = M_plus*(- M_L + 2)
        P_2bar_minus = M_minus*(- M_R - 2)
        D_plus = alpha_plus*(1 + beta_L) - beta_L*P_2bar_plus
        D_minus = alpha_minus*(1 + beta_R) - beta_R*P_2bar_minus
        
        F_P_p = np.array([0, D_plus*primitive_variables[2, i], 0])
        F_P_m = np.array([0, D_minus*primitive_variables[2, i + 1], 0])
        
        F[:, i] = F_C_p + F_P_p + F_C_m + F_P_m
        
#van_leer_1st_order_flux()

# ---------- ---------- ---------- ---------- ---------- ----------
# TRY TO REDUCE ROUND OFF ERROR
# ---------- ---------- ---------- ---------- ---------- ----------

conserved_variables = np.zeros((3, imax))

def primitive_to_conserved_variables():
    conserved_variables[0, :] = primitive_variables[0, cell_alias]
    conserved_variables[1, :] = primitive_variables[0, cell_alias]*primitive_variables[1, cell_alias]
    conserved_variables[2, :] = primitive_variables[2, cell_alias]/(gamma - 1) + 0.5*primitive_variables[0, cell_alias]*(primitive_variables[1, cell_alias]**2)
    
#primitive_to_conserved_variables()