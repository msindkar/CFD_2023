# AOE6145: Computational Fluid Dynamics, 2023
# Homework 2

import numpy as np
from hw_1_func import supersonic_nozzle_exact_solution
#import matplotlib as plt

p0 = 3E5 # Stagnation pressure, Pa
t0 = 600 # Stagnation temperature, K
gamma = 1.4

const1 = (gamma - 1)/2

# ---------- Set geometry ----------

print('Enter number of cells (even number):')
imax = int(input())

x_cell = np.arange(-1 + 1/imax, 1 + 1/imax, 2/imax)      # x-position of cell centers
x_intf = np.arange(-1, 1 + 1/imax, 2/imax)               # x-position of cell interfaces
x_intf[np.argmin(np.abs(x_intf))] = 0                    # set x = 0 at throat
A_cell = 0.2 + 0.4*(1 + np.sin(np.pi*(x_cell - 0.5)))    # area at cell centers
A_intf = 0.2 + 0.4*(1 + np.sin(np.pi*(x_intf - 0.5)))    # area at cell interfaces

cell_alias = list(range(1, imax + 1))     # Alias range for non-ghost cells easier indexing
intf_alias = list(range(1, imax + 2))     # Alias range for non-ghost interfaces easier indexing
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
interface_array = np.zeros((3, imax + 3)) # Array of interfaces, odd number, face at throat, 2 ghost cells

M = x_cell*1.4 + 1.6 # Mach number initial guess
for i in range(int(imax/2)):
    if M[i] >= 0.8:
        M[i] = 0.5

psi = 1 + const1*M**2