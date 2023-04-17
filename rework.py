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

M[0, cell_alias] = x_cell*1.4 + 1.6

psi = 1 + ((gamma - 1)/2)*M[0, cell_alias]**2
T[0, cell_alias] = t0/psi
primitive_variables[2, cell_alias] = p0/(psi**(gamma/(gamma - 1)))
primitive_variables[0, cell_alias] = primitive_variables[2, cell_alias]/(R_air*T[:, cell_alias])
a[0, cell_alias] = (gamma*R_air*T[0, cell_alias])**(0.5)
primitive_variables[1, cell_alias] = M[0, cell_alias]*a[0, cell_alias]

# ---------- ---------- ----------