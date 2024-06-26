# AOE6145: Computational Fluid Dynamics, 2023
# Homework 4

import numpy as np
from hw_1_func import supersonic_nozzle_exact_solution

p0 = 3E5 # Stagnation pressure, Pa
t0 = 600 # Stagnation temperature, K
kappa = 0
gamma = 1.4
R = 8314    # J/(kmol*K)
m_air = 28.96 # 
R_air = R/m_air # Specific gas constant for air
pback = 1.2E5

nmax = 0 # no. of iterations
cfl = 0.7

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

print('1 for isentropic case, anything else for shock case')
if int(input()) == 1:
    shock_flag = 0
    exact_solution = supersonic_nozzle_exact_solution(p0, t0, imax)
else:
    shock_flag = 1

primitive_variables = np.zeros((3, imax + 2))    # Array of cell centers, even number, 2 ghost cells, rho u P
cell_alias = list(range(1, imax + 1))     # Alias range for non-ghost cells easier indexing
M = np.zeros((1, imax + 2))
T = np.zeros((1, imax + 2))
a = np.zeros((1, imax + 2))
ht = np.zeros((1, imax + 2))

M[0, cell_alias] = x_cell*1.4 + 1.6

psi = 1 + ((gamma - 1)/2)*M[0, cell_alias]**2
T[0, cell_alias] = t0/psi
primitive_variables[2, cell_alias] = p0/(psi**(gamma/(gamma - 1)))
primitive_variables[0, cell_alias] = primitive_variables[2, cell_alias]/(R_air*T[:, cell_alias])
a[0, cell_alias] = (gamma*R_air*T[0, cell_alias])**(0.5)
primitive_variables[1, cell_alias] = M[0, cell_alias]*a[0, cell_alias]

ht[0, cell_alias] = ((gamma*R_air)/(gamma - 1))*T[0, cell_alias] + (primitive_variables[1, cell_alias]**2)/2

def upwind_boundary_conditions():
    M[0, 0] = 2*M[0, 1] - M[0, 2]
    
    psi_bc_0 = 1 + ((gamma - 1)/2)*M[0, 0]**2
    T[0, 0] = t0/psi_bc_0
    primitive_variables[2, 0] = p0/(psi_bc_0**(gamma/(gamma - 1)))
    primitive_variables[0, 0] = primitive_variables[2, 0]/(R_air*T[0, 0])
    a[0, 0] = (gamma*R_air*T[0, 0])**(1/2)
    primitive_variables[1, 0] = M[0, 0]*a[0, 0]
    
    ht[0, 0] = ((gamma*R_air)/(gamma - 1))*T[0, 0] + (primitive_variables[1, 0]**2)/2

    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]
    
    psi_bc_1 = 1 + ((gamma - 1)/2)*M[0, imax + 1]**2
    T[0, imax + 1] = t0/psi_bc_1
    if shock_flag == 0:
        primitive_variables[2, imax + 1] = p0/(psi_bc_1**(gamma/(gamma - 1)))
    else:
        primitive_variables[2, imax + 1] = 2*pback - primitive_variables[2, imax]
    primitive_variables[0, imax + 1] = primitive_variables[2, imax + 1]/(R_air*T[:, imax + 1])
    #a[0, imax + 1] = (gamma*R_air*T[0, imax + 1])**(1/2)
    a[0, imax + 1] = (gamma*primitive_variables[2, imax + 1]/primitive_variables[0, imax + 1])**(1/2)
    primitive_variables[1, imax + 1] = M[0, imax + 1]*a[0, imax + 1]
    
    ht[0, imax + 1] = ((gamma*R_air)/(gamma - 1))*T[0, imax + 1] + (primitive_variables[1, imax + 1]**2)/2

def upwind_boundary_conditions_try():
    M[0, 0] = 2*M[0, 1] - M[0, 2]
    
    psi_bc_0 = 1 + ((gamma - 1)/2)*M[0, 0]**2
    T[0, 0] = t0/psi_bc_0
    primitive_variables[2, 0] = p0/(psi_bc_0**(gamma/(gamma - 1)))
    primitive_variables[0, 0] = primitive_variables[2, 0]/(R_air*T[0, 0])
    a[0, 0] = (gamma*R_air*T[0, 0])**(1/2)
    primitive_variables[1, 0] = M[0, 0]*a[0, 0]
    
    ht[0, 0] = ((gamma*R_air)/(gamma - 1))*T[0, 0] + (primitive_variables[1, 0]**2)/2
    
    if shock_flag == 0:
        M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]
        psi_bc_1 = 1 + ((gamma - 1)/2)*M[0, imax + 1]**2
        T[0, imax + 1] = t0/psi_bc_1
        primitive_variables[2, imax + 1] = p0/(psi_bc_1**(gamma/(gamma - 1)))
        primitive_variables[0, imax + 1] = primitive_variables[2, imax + 1]/(R_air*T[:, imax + 1])
        a[0, imax + 1] = (gamma*primitive_variables[2, imax + 1]/primitive_variables[0, imax + 1])**(1/2)
        primitive_variables[1, imax + 1] = M[0, imax + 1]*a[0, imax + 1]
        ht[0, imax + 1] = ((gamma*R_air)/(gamma - 1))*T[0, imax + 1] + (primitive_variables[1, imax + 1]**2)/2

    else:
        primitive_variables[2, imax + 1] = 2*pback - primitive_variables[2, imax]
        T[0, imax + 1] = t0*(primitive_variables[2, imax + 1]/p0)**((gamma - 1)/gamma)
        primitive_variables[0, imax + 1] = primitive_variables[2, imax + 1]/(R_air*T[:, imax + 1])
        a[0, imax + 1] = (gamma*primitive_variables[2, imax + 1]/primitive_variables[0, imax + 1])**(1/2)
        M[0, imax + 1] = (((t0/T[:, imax + 1]) - 1)*(2/(gamma - 1)))**0.5
        primitive_variables[1, imax + 1] = M[0, imax + 1]*a[0, imax + 1]
        
def set_boundary_conditions():
    M[0, 0] = 2*M[0, 1] - M[0, 2]
    primitive_variables[:, 0] = 2*primitive_variables[:, 1] - primitive_variables[:, 2]
    T[:, 0] = 2*T[:, 1] - T[:, 2]
    a[:, 0] = 2*a[:, 1] - a[:, 2]
    
    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]
    primitive_variables[(0, 1), imax + 1] = 2*primitive_variables[(0, 1), imax] - primitive_variables[(0, 1), imax - 1]
    T[:, imax + 1] = 2*T[:, imax] - T[:, imax - 1]
    a[:, imax + 1] = 2*a[:, imax] - a[:, imax - 1]
    
    if shock_flag == 0:
        primitive_variables[2, imax + 1] = 2*primitive_variables[2, imax] - primitive_variables[2, imax - 1]
    else:
        primitive_variables[2, imax + 1] = 2*pback - primitive_variables[2, imax]
    

F = np.zeros((3, imax + 1))

def van_leer_1st_order_flux():
    for i in range(imax + 1):
        M_L = primitive_variables[1, i]/a[0, i]
        M_R = primitive_variables[1, i + 1]/a[0, i + 1]
        M_plus = (1/4)*(M_L + 1)**2
        M_minus = -(1/4)*(M_R - 1)**2
        beta_L = -max(0, (1 - int(M_L)))
        beta_R = -max(0, (1 - int(M_R)))
        alpha_plus = (1/2)*(1 + np.sign(M_L)) #ABS of M???
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

lam     = np.zeros((1, 3))
r_eig   = np.zeros((3, 3))
d_w     = np.zeros((1, 3))
fi      = np.zeros((3, 1))
fi1     = np.zeros((3, 1))

def roe_1st_order_flux():
    for i in range(imax + 1):
        roe_R   = (primitive_variables[0, i + 1]/primitive_variables[0, i])**(1/2)
        roe_rho = roe_R*primitive_variables[0, i]
        roe_u   = (roe_R*primitive_variables[1, i + 1] + primitive_variables[1, i])/(roe_R + 1)
        roe_ht  = (roe_R*ht[0, i + 1] + ht[0, i])/(roe_R + 1)
        roe_a2  = (gamma - 1)*(roe_ht - (roe_u**2)/2)
        k = 0
        lam[0, 0]     = roe_u
        lam[0, 1]     = roe_u + (roe_a2)**(0.5)
        lam[0, 2]     = roe_u - (roe_a2)**(0.5)
        lam_p   = np.zeros((1, 3))
        lam_m   = np.zeros((1, 3))
        for o in lam[0, :]<0:
            if o:
                lam_m[0, k] = lam[0, k]
            else:
                lam_p[0, k] = lam[0, k]
            k+=1
        r_eig[:, 0] = np.array([1, roe_u, (roe_u**2)/2]).transpose()
        r_eig[:, 1] =  (roe_rho/(2*(roe_a2**0.5)))*np.array([1, roe_u + (roe_a2)**(0.5), roe_ht + roe_u*(roe_a2)**(0.5)])
        r_eig[:, 2] = -(roe_rho/(2*(roe_a2**0.5)))*np.array([1, roe_u - (roe_a2)**(0.5), roe_ht - roe_u*(roe_a2)**(0.5)])
        d_w  [0, 0] = (primitive_variables[0, i + 1] - primitive_variables[0, i]) - (primitive_variables[2, i + 1] - primitive_variables[2, i])/roe_a2
        d_w  [0, 1] = (primitive_variables[1, i + 1] - primitive_variables[1, i]) + (primitive_variables[2, i + 1] - primitive_variables[2, i])/(roe_rho*(roe_a2**0.5))
        d_w  [0, 2] = (primitive_variables[1, i + 1] - primitive_variables[1, i]) - (primitive_variables[2, i + 1] - primitive_variables[2, i])/(roe_rho*(roe_a2**0.5))
        fi[:, 0]    = np.array([primitive_variables[0, i]*primitive_variables[1, i], primitive_variables[0, i]*primitive_variables[1, i]**2 + primitive_variables[2, i], primitive_variables[0, i]*primitive_variables[1, i]*ht[0, i]])
        fi1[:, 0]   = np.array([primitive_variables[0, i + 1]*primitive_variables[1, i + 1], primitive_variables[0, i + 1]*primitive_variables[1, i + 1]**2 + primitive_variables[2, i + 1], primitive_variables[0, i + 1]*primitive_variables[1, i + 1]*ht[0, i + 1]])
        sigma   = np.zeros((3, 1))
        for q in range(3):
            sigma[:, 0] = sigma[:, 0] + (lam_p[0, q] - lam_m[0, q])*d_w[0, q]*r_eig[:, q]
        F[:, i] = 0.5*(fi[:, 0] + fi1[:, 0]) - 0.5*sigma[:, 0]

# primitive_variables_e = np.zeros((3, 2))
# M_e = np.zeros((1, 2))
# T_e = np.zeros((1, 2))

def extrapolate_for_2nd_order():    
    M_e[0, 0] = 2*M[0, 0] - M[0, 1]
    psi_bc_0_e = 1 + ((gamma - 1)/2)*M_e[0, 0]**2
    T_e[0, 0] = t0/psi_bc_0_e
    primitive_variables_e[2, 0] = p0/(psi_bc_0_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, 0] = primitive_variables_e[2, 0]/(R_air*T_e[0, 0])
    primitive_variables_e[1, 0] = M_e[0, 0]*(gamma*R_air*T_e[0, 0])**(0.5)
    
    
    M_e[0, 1] = 2*M[0, imax + 1] - M[0, imax]
    psi_bc_1_e = 1 + ((gamma - 1)/2)*M_e[0, 1]**2
    T_e[0, 1] = t0/psi_bc_1_e
    primitive_variables_e[2, 1] = p0/(psi_bc_1_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, 1] = primitive_variables_e[2, 1]/(R_air*T_e[0, 1])
    primitive_variables_e[1, 1] = M_e[0, 1]*(gamma*R_air*T_e[0, 1])**(0.5)
    
# r_plus = np.zeros((4, imax + 1))
# r_minus = np.zeros((4, imax + 1))
# r_den = np.zeros((4, imax + 1))
# psi_plus = np.zeros((4, imax + 1))
# psi_minus = np.zeros((4, imax + 1))

def van_leer_limiter():
    for i in range(imax + 1):
        if i == 0:
            r_den[[0, 1, 2], 0] = primitive_variables[:, 1] - primitive_variables[:, 0]
            r_den[[3], 0] = T[0, 1] - T[0, 0]
            
            for g in range(0, 4):
                if abs(r_den[g, 0]) < 1E-6:
                    r_den[g, 0] = np.sign(r_den[g, 0])*1E-6
            r_plus[[0, 1, 2], 0] = (primitive_variables[:, 2] - primitive_variables[:, 1])/r_den[[0, 1, 2], 0]
            r_plus[3, 0] = (T[0, 2] - T[0, 1])/r_den[3, 0]
            
            r_minus[[0, 1, 2], 0] = (primitive_variables[:, 0] - primitive_variables_e[:, 0])/r_den[[0, 1, 2], 0]
            r_minus[3, 0] = (T[0, 0] - T_e[0, 0])/r_den[3, 0]
            
        elif i == imax:
            r_den[[0, 1, 2], imax] = primitive_variables[:, imax + 1] - primitive_variables[:, imax]
            r_den[[3], imax] = T[0, imax + 1] - T[0, imax]

            for g in range(0, 4):
                if abs(r_den[g, imax]) < 1E-6:
                    r_den[g, imax] = np.sign(r_den[g, imax])*1E-6
            r_plus[[0, 1, 2], imax] = (primitive_variables_e[:, 1] - primitive_variables[:, imax + 1])/r_den[[0, 1, 2], imax]
            r_plus[3, imax] = (T_e[0, 1] - T[0, imax + 1])/r_den[3, imax]

            
            r_minus[[0, 1, 2], imax] = (primitive_variables[:, imax] - primitive_variables[:, imax - 1])/r_den[[0, 1, 2], imax]
            r_minus[3, imax] = (T[0, imax] - T[0, imax - 1])/r_den[3, imax]
            
        else:
            r_den[[0, 1, 2], i] = primitive_variables[:, i + 1] - primitive_variables[:, i]
            r_den[[3], i] = T[0, i + 1] - T[0, i]

            for g in range(0, 4):
                if abs(r_den[g, i]) < 1E-6:
                    r_den[g, i] = np.sign(r_den[g, i])*1E-6
            r_plus[[0, 1, 2], i] = (primitive_variables[:, i + 2] - primitive_variables[:, i + 1])/r_den[[0, 1, 2], i]
            r_plus[3, i] = (T[0, i + 2] - T[0, i + 1])/r_den[3, i]
            
            r_minus[[0, 1, 2], i] = (primitive_variables[:, i] - primitive_variables[:, i - 1])/r_den[[0, 1, 2], i]
            r_minus[3, i] = (T[0, i] - T[0, i - 1])/r_den[3, i]
        
        psi_plus[:, i] = (r_plus[:, i] + np.abs(r_plus[:, i]))/(1 + r_plus[:, i])
        psi_minus[:, i] = (r_minus[:, i] + np.abs(r_minus[:, i]))/(1 + r_minus[:, i])

# prim_L = np.zeros((3, imax + 1))
# prim_R = np.zeros((3, imax + 1))
# T_L    = np.zeros((1, imax + 1))
# T_R    = np.zeros((1, imax + 1))
# epsilon= 0

def compute_LR_states_2nd_order():
    for i in range(imax + 1):
        if i == 0:
            prim_L[:, 0] = primitive_variables[:, 0]
            T_L[0, 0] = T[0, 0]
            prim_R[:, 0] = primitive_variables[:, 1] - (epsilon/4)*((1 - kappa)*psi_minus[[0, 1, 2], 1]*(primitive_variables[:, 2] - primitive_variables[:, 1]) + (1 + kappa)*psi_plus[[0, 1, 2], 0]*(primitive_variables[:, 1] - primitive_variables[:, 0]))
            T_R[:, 0] = T[:, 1] - (epsilon/4)*((1 - kappa)*psi_minus[3, 1]*(T[:, 2] - T[:, 1]) + (1 + kappa)*psi_plus[3, 0]*(T[:, 1] - T[:, 0]))
        elif i == imax:
            prim_R[:, imax] = primitive_variables[:, imax + 1]
            T_R[0, imax] = T[0, imax + 1]
            prim_L[:, imax] = primitive_variables[:, imax] + (epsilon/4)*((1 - kappa)*psi_plus[[0, 1, 2], imax - 1]*(primitive_variables[:, imax] - primitive_variables[:, imax - 1]) + (1 + kappa)*psi_minus[[0, 1, 2], imax]*(primitive_variables[:, imax + 1] - primitive_variables[:, imax]))
            T_L[:, imax] = T[:, imax] + (epsilon/4)*((1 - kappa)*psi_plus[3, imax - 1]*(T[:, imax] - T[:, imax - 1]) + (1 + kappa)*psi_minus[3, imax]*(T[:, imax + 1] - T[:, imax]))
        else:
            prim_L[:, i] = primitive_variables[:, i] + (epsilon/4)*((1 - kappa)*psi_plus[[0, 1, 2], i - 1]*(primitive_variables[:, i] - primitive_variables[:, i - 1]) + (1 + kappa)*psi_minus[[0, 1, 2], i]*(primitive_variables[:, i + 1] - primitive_variables[:, i]))
            T_L[:, i] = T[:, i] + (epsilon/4)*((1 - kappa)*psi_plus[3, i - 1]*(T[:, i] - T[:, i - 1]) + (1 + kappa)*psi_minus[3, i]*(T[:, i + 1] - T[:, i]))
            prim_R[:, i] = primitive_variables[:, i + 1] - (epsilon/4)*((1 - kappa)*psi_minus[[0, 1, 2], i + 1]*(primitive_variables[:, i + 2] - primitive_variables[:, i + 1]) + (1 + kappa)*psi_plus[[0, 1, 2], i]*(primitive_variables[:, i + 1] - primitive_variables[:, i]))
            T_R[:, i] = T[:, i + 1] - (epsilon/4)*((1 - kappa)*psi_minus[3, i + 1]*(T[:, i + 2] - T[:, i + 1]) + (1 + kappa)*psi_plus[3, i]*(T[:, i + 1] - T[:, i]))

primitive_variables_e = np.zeros((3, imax + 6))
M_e = np.zeros((1, imax + 6))
T_e = np.zeros((1, imax + 6))

r_plus = np.zeros((4, 1))
r_minus = np.zeros((4, 1))
r_den = np.zeros((4, 1))
psi_plus = np.zeros((4, imax + 3))
psi_minus = np.zeros((4, imax + 3))

prim_L = np.zeros((3, imax + 1))
prim_R = np.zeros((3, imax + 1))
T_L    = np.zeros((1, imax + 1))
T_R    = np.zeros((1, imax + 1))
epsilon= 0

def extrapolate_for_2nd_order_ePsi():
    M_e[0, 1] = 2*M[0, 0] - M[0, 1]
    psi_bc_0_e = 1 + ((gamma - 1)/2)*M_e[0, 1]**2
    T_e[0, 1] = t0/psi_bc_0_e
    primitive_variables_e[2, 1] = p0/(psi_bc_0_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, 1] = primitive_variables_e[2, 1]/(R_air*T_e[0, 1])
    primitive_variables_e[1, 1] = M_e[0, 1]*(gamma*R_air*T_e[0, 1])**(0.5)
    
    M_e[0, 0] = 2*M_e[0, 1] - M[0, 0]
    psi_bc_0_e = 1 + ((gamma - 1)/2)*M_e[0, 0]**2
    T_e[0, 0] = t0/psi_bc_0_e
    primitive_variables_e[2, 0] = p0/(psi_bc_0_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, 0] = primitive_variables_e[2, 0]/(R_air*T_e[0, 0])
    primitive_variables_e[1, 0] = M_e[0, 0]*(gamma*R_air*T_e[0, 0])**(0.5)
    
    M_e[0, imax + 4] = 2*M[0, imax + 1] - M[0, imax]
    psi_bc_1_e = 1 + ((gamma - 1)/2)*M_e[0, imax + 4]**2
    T_e[0, imax + 4] = t0/psi_bc_1_e
    primitive_variables_e[2, imax + 4] = p0/(psi_bc_1_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, imax + 4] = primitive_variables_e[2, imax + 4]/(R_air*T_e[0, imax + 4])
    primitive_variables_e[1, imax + 4] = M_e[0, imax + 4]*(gamma*R_air*T_e[0, imax + 4])**(0.5)
    
    M_e[0, imax + 5] = 2*M_e[0, imax + 4] - M[0, imax + 1]
    psi_bc_1_e = 1 + ((gamma - 1)/2)*M_e[0, imax + 5]**2
    T_e[0, imax + 5] = t0/psi_bc_1_e
    primitive_variables_e[2, imax + 5] = p0/(psi_bc_1_e**(gamma/(gamma - 1)))
    primitive_variables_e[0, imax + 5] = primitive_variables_e[2, imax + 5]/(R_air*T_e[0, imax + 5])
    primitive_variables_e[1, imax + 5] = M_e[0, imax + 5]*(gamma*R_air*T_e[0, imax + 5])**(0.5)
    
    primitive_variables_e[:, 2:imax + 4] = primitive_variables[:, :]
    T_e[:, 2:imax + 4] = T[:, :]

def van_leer_limiter_ePsi():
    for i in range(imax + 3):
        h = i + 1
        r_den[[0, 1, 2], 0] = primitive_variables_e[:, h + 1] - primitive_variables_e[:, h]
        r_den[3] = T_e[:, h + 1] - T_e[:, h]
        
        for g in range(0, 4):
            if abs(r_den[g]) < 1E-6:
                r_den[g] = np.sign(r_den[g])*1E-6
        
        r_plus[[1, 2, 3], 0] = (primitive_variables_e[:, h + 2] - primitive_variables_e[:, h + 1])/r_den[[0, 1, 2], 0]
        r_plus[3] = (T_e[0, h + 2] - T_e[0, h + 1])/r_den[3]
        
        r_minus[[1, 2, 3], 0] = (primitive_variables_e[:, h] - primitive_variables_e[:, h - 1])/r_den[[0, 1, 2], 0]
        r_plus[3] = (T_e[0, h] - T_e[0, h - 1])/r_den[3]
        
        psi_plus[:, i] = (r_plus[:, 0] + np.abs(r_plus[:, 0]))/(1 + r_plus[:, 0])
        psi_minus[:, i] = (r_minus[:, 0] + np.abs(r_minus[:, 0]))/(1 + r_minus[:, 0])
            
def compute_LR_states_2nd_order_ePsi():
    ''
            
def van_leer_2nd_order_flux():
    extrapolate_for_2nd_order()
    van_leer_limiter()
    compute_LR_states_2nd_order()
    for i in range(imax + 1):
        M_L = prim_L[1, i]/(abs(gamma*prim_L[2, i]/prim_L[0, i]))**0.5
        M_R = prim_R[1, i]/(abs(gamma*prim_R[2, i]/prim_R[0, i]))**0.5
        M_plus = (1/4)*(M_L + 1)**2
        M_minus = -(1/4)*(M_R - 1)**2
        beta_L = -max(0, (1 - int(M_L)))
        beta_R = -max(0, (1 - int(M_R)))
        alpha_plus = (1/2)*(1 + np.sign(M_L))
        alpha_minus = (1/2)*(1 - np.sign(M_R))
        c_plus = alpha_plus*(1 + beta_L)*M_L - beta_L*M_plus
        c_minus = alpha_minus*(1 + beta_R)*M_R - beta_R*M_minus

        ht_L = ((gamma*R_air)/(gamma - 1))*T_L[0, i] + (prim_L[1, i]**2)/2
        ht_R = ((gamma*R_air)/(gamma - 1))*T_R[0, i] + (prim_R[1, i]**2)/2

        F_C_p = prim_L[0, i]*((abs(gamma*prim_L[2, i]/prim_L[0, i]))**0.5)*c_plus*np.array([1, prim_L[1, i], ht_L])
        F_C_m = prim_R[0, i]*((abs(gamma*prim_R[2, i]/prim_R[0, i]))**0.5)*c_minus*np.array([1, prim_R[1, i], ht_R])
        
        P_2bar_plus = M_plus*(- M_L + 2)
        P_2bar_minus = M_minus*(- M_R - 2)
        D_plus = alpha_plus*(1 + beta_L) - beta_L*P_2bar_plus
        D_minus = alpha_minus*(1 + beta_R) - beta_R*P_2bar_minus
        
        F_P_p = np.array([0, D_plus*prim_L[2, i], 0])
        F_P_m = np.array([0, D_minus*prim_R[2, i], 0])
        
        F[:, i] = F_C_p + F_P_p + F_C_m + F_P_m
        
conserved_variables = np.zeros((3, imax))

def roe_2nd_order_flux():
    extrapolate_for_2nd_order()
    van_leer_limiter()
    compute_LR_states_2nd_order()
    for i in range(imax + 1):
        roe_R   = (prim_R[0, i]/prim_L[0, i])**(1/2)
        roe_rho = roe_R*prim_L[0, i]
        roe_u   = (roe_R*prim_R[1, i] + prim_L[1, i])/(roe_R + 1)
        
        ht_L = ((gamma*R_air)/(gamma - 1))*T_L[0, i] + (prim_L[1, i]**2)/2
        ht_R = ((gamma*R_air)/(gamma - 1))*T_R[0, i] + (prim_R[1, i]**2)/2
        
        roe_ht  = (roe_R*ht_R + ht_L)/(roe_R + 1)
        roe_a2  = (gamma - 1)*(roe_ht - (roe_u**2)/2)
        k = 0
        lam[0, 0]     = roe_u
        lam[0, 1]     = roe_u + (roe_a2)**(0.5)
        lam[0, 2]     = roe_u - (roe_a2)**(0.5)
        lam_p   = np.zeros((1, 3))
        lam_m   = np.zeros((1, 3))
        for o in lam[0, :]<0:
            if o:
                lam_m[0, k] = lam[0, k]
            else:
                lam_p[0, k] = lam[0, k]
            k+=1
        r_eig[:, 0] = np.array([1, roe_u, (roe_u**2)/2]).transpose()
        r_eig[:, 1] =  (roe_rho/(2*(roe_a2**0.5)))*np.array([1, roe_u + (roe_a2)**(0.5), roe_ht + roe_u*(roe_a2)**(0.5)])
        r_eig[:, 2] = -(roe_rho/(2*(roe_a2**0.5)))*np.array([1, roe_u - (roe_a2)**(0.5), roe_ht - roe_u*(roe_a2)**(0.5)])
        d_w  [0, 0] = (prim_R[0, i] - prim_L[0, i]) - (prim_R[2, i] - prim_L[2, i])/roe_a2
        d_w  [0, 1] = (prim_R[1, i] - prim_L[1, i]) + (prim_R[2, i] - prim_L[2, i])/(roe_rho*(roe_a2**0.5))
        d_w  [0, 2] = (prim_R[1, i] - prim_L[1, i]) - (prim_R[2, i] - prim_L[2, i])/(roe_rho*(roe_a2**0.5))
        fi[:, 0]    = np.array([prim_L[0, i]*prim_L[1, i], prim_L[0, i]*prim_L[1, i]**2 + prim_L[2, i], prim_L[0, i]*prim_L[1, i]*ht_L])
        fi1[:, 0]   = np.array([prim_R[0, i]*prim_R[1, i], prim_R[0, i]*prim_R[1, i]**2 + prim_R[2, i], prim_R[0, i]*prim_R[1, i]*ht_R])
        sigma   = np.zeros((3, 1))
        for q in range(3):
            sigma[:, 0] = sigma[:, 0] + (lam_p[0, q] - lam_m[0, q])*d_w[0, q]*r_eig[:, q]
        F[:, i] = 0.5*(fi[:, 0] + fi1[:, 0]) - 0.5*sigma[:, 0]


def primitive_to_conserved_variables():
    conserved_variables[0, :] = primitive_variables[0, cell_alias]
    conserved_variables[1, :] = primitive_variables[0, cell_alias]*primitive_variables[1, cell_alias]
    conserved_variables[2, :] = primitive_variables[2, cell_alias]/(gamma - 1) + 0.5*primitive_variables[0, cell_alias]*(primitive_variables[1, cell_alias]**2)
    
primitive_to_conserved_variables()

def conserved_to_primitive_variables():
    primitive_variables[0, cell_alias] = conserved_variables[0, :]
    primitive_variables[1, cell_alias] = conserved_variables[1, :]/conserved_variables[0, :]
    primitive_variables[2, cell_alias] = (gamma - 1)*conserved_variables[2, :] - 0.5*(gamma - 1)*(conserved_variables[1, :]**2)/(conserved_variables[0, :])

def update_domain_variables():
    T[0, cell_alias] = primitive_variables[2, cell_alias]/(R_air*primitive_variables[0, cell_alias])
    a[0, cell_alias] = (gamma*R_air*T[0, cell_alias])**(1/2)
    M[0, cell_alias] = primitive_variables[1, cell_alias]/a[0, cell_alias]
    ht[0, cell_alias] = ((gamma*R_air)/(gamma - 1))*T[0, cell_alias] + (primitive_variables[1, cell_alias]**2)/2

S = np.zeros((3, imax))

def source_terms():
    S[1, :] = primitive_variables[2, cell_alias]*dA_dx
    
source_terms()

dt = np.zeros((1, imax))
lambda_max = np.zeros((1, imax))

def compute_time_step():
    lambda_max[0, :] = np.abs(primitive_variables[1, cell_alias]) + a[0, cell_alias]
    dt[0, :] = cfl*(dx/lambda_max)
    
compute_time_step()

f = open('soln.dat', 'w')
f.write('TITLE = "Quasi-1D Nozzle Solution"\n')
f.write('variables="x(m)""Area(m^2)""Mach""rho(kg/m^3)""u(m/s)""Press(N/m^2)""T(K)"\n')

f1 = open('res.dat', 'w')
f1.write('TITLE = "Quasi-1D Nozzle Residuals"\n')
f1.write('variables="Iteration""Coninuity""X-momentum""Energy"\n')

def write_to_file():
    f.write('zone T="' + str(j) + '"\n')
    f.write('I=' + str(imax) + '\n')
    f.write('DATAPACKING=POINT\n')
    f.write('DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)\n')
    for g in range(imax):
        f.write(str(float(x_cell[g]))  + " " +  str(float(A_cell[g]))  + " " +  str(float(M[0, g + 1]))  + " " +  str(float(primitive_variables[0, g + 1]))  + " " +  str(float(primitive_variables[1, g + 1]))  + " " +  str(float(primitive_variables[2, g + 1]))  + " " +  str(float(T[0, g + 1])) + '\n')
    print('Wrote converged')
    
def write_exact():
    f4 = open('exact.dat', 'w')
    f4.write('TITLE = "Quasi-1D Nozzle Exact Solution"\n')
    f4.write('variables="x(m)""Area(m^2)""Mach""rho(kg/m^3)""u(m/s)""Press(N/m^2)""T(K)"\n')
    f4.write('zone T="' + str(j) + '"\n')
    f4.write('I=' + str(imax) + '\n')
    f4.write('DATAPACKING=POINT\n')
    f4.write('DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)\n')
    for g in range(imax):
        f4.write(str(float(exact_solution[0, :][g]))  + " " +  str(float(exact_solution[1, :][g]))  + " " +  str(float(exact_solution[2, :][g]))  + " " +  str(float(exact_solution[5, :][g]))  + " " +  str(float(exact_solution[6, :][g]))  + " " +  str(float(exact_solution[4, :][g]))  + " " +  str(float(exact_solution[3, :][g])) + '\n')
    f4.close()
    print('Wrote exact')

init_norm = np.zeros((3, 1))
R_i = np.zeros((3, imax))
norm = np.zeros((3, 1))
res = np.zeros((3, 1))

upwind_boundary_conditions_try()

van_leer_1st_order_flux()
#roe_1st_order_flux()

for i in range(imax):
    R_i[:, i] = F[:, i + 1]*A_intf[i + 1] - F[:, i]*A_intf[i] - S[:, i]*dx
    
init_norm[0] = ((np.sum(R_i[0, :]**2))/imax)**0.5 # continuity
init_norm[1] = ((np.sum(R_i[1, :]**2))/imax)**0.5 # x-momentum
init_norm[2] = ((np.sum(R_i[2, :]**2))/imax)**0.5 # energy

def out_steady_state_iterative_residuals():
    
    norm[0] = ((np.sum(R_i[0, :]**2))/imax)**0.5 # continuity
    norm[1] = ((np.sum(R_i[1, :]**2))/imax)**0.5 # x-momentum
    norm[2] = ((np.sum(R_i[2, :]**2))/imax)**0.5 # energy
    
    res[0] = norm[0]/init_norm[0]
    res[1] = norm[1]/init_norm[1]
    res[2] = norm[2]/init_norm[2]
    
    if j%100 == 0:
        f1.write(str(j) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])) + '\n')
        print(str(j) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])))
        
def de_norms():
    DE = np.zeros((3, imax))
    DE_norm = np.zeros((3, 1))
    
    DE[0, :] = exact_solution[5, :] -  primitive_variables[0, cell_alias]
    DE[1, :] = exact_solution[6, :] -  primitive_variables[1, cell_alias]
    DE[2, :] = exact_solution[4, :] -  primitive_variables[2, cell_alias]
    
    DE_norm[0] = ((np.sum(DE[0, :]**2))/imax)**0.5
    DE_norm[1] = ((np.sum(DE[1, :]**2))/imax)**0.5
    DE_norm[2] = ((np.sum(DE[2, :]**2))/imax)**0.5

    f3 = open('DE_norms.txt', 'w')
    print('DE Norms:\n' + 'rho = ' + str(float(DE_norm[0])) + ' ' + 'u = ' + str(float(DE_norm[1])) + ' ' + 'P = ' + str(float(DE_norm[2])))
    f3.write('DE Norms:\n' + 'dx =' + str(dx) + 'rho = ' + str(float(DE_norm[0])) + ' ' + 'u = ' + str(float(DE_norm[1])) + ' ' + 'P = ' + str(float(DE_norm[2])))
    f3.close()

print('Iteration' + " " + 'Continuity' + " " + 'X - mtm' + " " + 'Energy')

for j in range(nmax + 1):
    # if j == 1000:
    #     epsilon = 1
    #     cfl = 0.4
    #     print('---------- Switched to 2nd order ----------')
    #     # USE THIS TO SWITCH TO 2ND ORDER, EDIT 'j' to change when the switch happens
    #upwind_boundary_conditions()
    upwind_boundary_conditions_try()
    #set_boundary_conditions()
    van_leer_1st_order_flux()
    #van_leer_2nd_order_flux()
    #roe_1st_order_flux()
    #roe_2nd_order_flux()
    compute_time_step()
    source_terms()
    primitive_to_conserved_variables()
    for i in range(imax):
        R_i[:, i] = F[:, i + 1]*A_intf[i + 1] - F[:, i]*A_intf[i] - S[:, i]*dx
    conserved_variables = conserved_variables - R_i*(dt/V)
    conserved_to_primitive_variables()
    update_domain_variables()
    out_steady_state_iterative_residuals()
    if (res[:, :] <= 1E-8).all():
        print('Solution converged in ' + str(j) + ' iterations')
        write_to_file()
        break
    
if shock_flag == 0:
    de_norms()
    write_exact()
write_to_file()
f.close()
f1.close()