# AOE6145: Computational Fluid Dynamics, 2023
# Homework 2

import numpy as np
from hw_1_func import supersonic_nozzle_exact_solution
#import matplotlib as plt

p0 = 3E5 # Stagnation pressure, Pa
t0 = 600 # Stagnation temperature, K
gamma = 1.4
R = 8314    # J/(kmol*K)
m_air = 28.96 # 
R_air = R/m_air # Specific gas constant for air

const1 = (gamma - 1)/2
const2 = gamma/(gamma - 1)

# ---------- Set geometry ----------

print('Enter number of cells (even number):')
imax = int(input())

x_cell = np.arange(-1 + 1/imax, 1 + 1/imax, 2/imax)      # x-position of cell centers
x_intf = np.arange(-1, 1 + 1/imax, 2/imax)               # x-position of cell interfaces
x_intf[np.argmin(np.abs(x_intf))] = 0                    # set x = 0 at throat
A_cell = 0.2 + 0.4*(1 + np.sin(np.pi*(x_cell - 0.5)))    # area at cell centers
A_intf = 0.2 + 0.4*(1 + np.sin(np.pi*(x_intf - 0.5)))    # area at cell interfaces
dA_dx  = 0.4*np.pi*np.cos(np.pi*x *np.pi*0.5)

# ---------- Check case ----------

print('1 for isentropic case, anything else for shock case')
if int(input()) == 1:
    shock_flag = 0
    exact_solution = supersonic_nozzle_exact_solution(p0, t0, imax)
else:
    shock_flag = 1
    pback = 1.2E5 # Back pressure for shock case, Pa
    
# ---------- Set inital conditions ----------

center_array = np.zeros((3, imax + 2))    # Array of cell centers, even number, 2 ghost cells
# interface_array = np.zeros((3, imax + 3)) # Array of interfaces, odd number, face at throat, 2 ghost cells
cell_alias = list(range(1, imax + 1))     # Alias range for non-ghost cells easier indexing
intf_alias = list(range(1, imax + 2))     # Alias range for non-ghost interfaces easier indexing

M = np.zeros((1, imax + 2))
p = np.zeros((1, imax + 2))
M[0, cell_alias] = x_cell*1.4 + 1.6 # Mach number initial guess
for i in range(int((imax + 2)/2)):
    if M[0, i] >= 0.8:
        M[0, i] = 0.5

psi = 1 + const1*M[0, cell_alias]**2
center_array[2, cell_alias] = t0/psi                                                            # Set initial temperature
center_array[1, cell_alias] = M[0, cell_alias]*np.sqrt(gamma*R_air*center_array[2, cell_alias]) # Set initial velocity
p[0, cell_alias] = p0/psi**const2                                                               # Set initial pressure
center_array[0, cell_alias] = p[0, cell_alias]/(R_air*center_array[2, cell_alias])              # Set initial density

# ---------- Set boundary conditions ----------

def set_boundary_conditions():
    center_array[:, 0] = 2*center_array[:, 1] - center_array[:, 2]
    p[0, 0] = 2*p[0, 1] - p[0, 2]
    M[0, 0] = 2*M[0, 1] - M[0, 2]                                   # left BCs by extrapolation
    if M[0, 0] <= 0.11668889438289902/100:
        M[0, 0] = 0.11668889438289902/100
        print('---------- Corrected left extrapolated Mach number ----------')
        
    center_array[:, imax + 1] = 2*center_array[:, imax] - center_array[:, imax - 1]
    if shock_flag == 0:
        p[0, imax + 1] = 2*p[0, imax] - p[0, imax - 1]
    else:
        p[0, imax + 1] = 2*pback  - p[0, imax - 1]
    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]                  # right BCs by extrapolation
    if M[0, 0] <= 0.11668889438289902/100:
        M[0, 0] = 0.11668889438289902/100
        print('---------- Corrected right extrapolated Mach number ----------')
        
    # possibly apply Mach limiter to whole domain?
    
set_boundary_conditions()

# ---------- Construct conserved and flux vector ------------

U = np.zeros(3, imax)
U

# ---------- Check iterative convergence ----------

def check_iterative_convergence():
    "bruh"