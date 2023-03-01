# AOE6145: Computational Fluid Dynamics, 2023
# Homework 2
#

import numpy as np
from hw_1_func import *
#import matplotlib as plt

p0 = 3E5 # Stagnation pressure, Pa
t0 = 600 # Stagnation temperature, K
pback = 1.2E5 # Back pressure for shock case, Pa

# ---------- Set geometry ----------

print('Enter number of cells (even number):')
imax = int(input())

print('1 for isentropic case, anything else for shock case')
if int(input()) == 1:
    shock_flag = 0
else:
    shock_flag = 1

center_array = np.zeros((1, imax + 2))                   # Array of cell centers, even number, 2 ghost cells
interface_array = np.zeros((1, imax + 3))                # Array of interfaces, odd number, face at throat, 2 ghost cells

