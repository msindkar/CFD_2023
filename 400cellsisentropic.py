# AOE6145: Computational Fluid Dynamics, 2023
# Homework 2

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

const1 = (gamma - 1)/2
const2 = gamma/(gamma - 1)
const3 = 2/(gamma - 1)

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

center_array = np.zeros((3, imax + 2))    # Array of cell centers, even number, 2 ghost cells
cell_alias = list(range(1, imax + 1))     # Alias range for non-ghost cells easier indexing
intf_alias = list(range(1, imax + 2))     # Alias range for non-ghost interfaces easier indexing

M = np.zeros((1, imax + 2))
p = np.zeros((1, imax + 2))
M[0, cell_alias] = x_cell*1.4 + 1.6 # Mach number initial guess
#M[0, cell_alias] = x_cell*0.9 + 1
# for i in range(int((imax + 2)/2)):
#     if M[0, i] >= 1:
#         M[0, i] = 0.99

psi                         = 1 + const1*M[0, cell_alias]**2
center_array[2, cell_alias] = t0/psi                                                            # Set initial temperature
center_array[1, cell_alias] = M[0, cell_alias]*np.sqrt(gamma*R_air*center_array[2, cell_alias]) # Set initial velocity
p[0, cell_alias]            = p0/psi**const2                                                    # Set initial pressure
center_array[0, cell_alias] = p[0, cell_alias]/(R_air*center_array[2, cell_alias])              # Set initial density

# ---------- Set boundary conditions ----------
# j = 0
def set_boundary_conditions():
    center_array[:, 0] = 2*center_array[:, 1] - center_array[:, 2]
    p[0, 0] = 2*p[0, 1] - p[0, 2]
    M[0, 0] = 2*M[0, 1] - M[0, 2]                                   # left BCs by extrapolation
        
    center_array[:, imax + 1] = 2*center_array[:, imax] - center_array[:, imax - 1]
    if shock_flag == 0:
        p[0, imax + 1] = 2*p[0, imax] - p[0, imax - 1]
    else:
        p[0, imax + 1] = 2*pback  - p[0, imax]
    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]                  # right BCs by extrapolation
    
    M[0, :][M[0, :] < 0.11668889438289902/100] = 0.11668889438289902/100
    center_array[1, :][center_array[1, :] < 57.2706378650403/100] = 57.2706378650403/100
    
set_boundary_conditions()

def try_boundary_conditions():
    M[0, 0] = 2*M[0, 1] - M[0, 2] 
    M[0, imax + 1] = 2*M[0, imax] - M[0, imax - 1]
    
    psi1 = 1 + const1*M[0, 0]**2
    center_array[2, 0] = t0/psi1                                                            # Set initial temperature
    center_array[1, 0] = M[0, 0]*np.sqrt(gamma*R_air*center_array[2, 0])                    # Set initial velocity
    p[0, 0]            = p0/psi1**const2                                                    # Set initial pressure
    center_array[0, 0] = p[0, 0]/(R_air*center_array[2, 0])
    
    psi2 = 1 + const1*M[0, imax + 1]**2
    center_array[2, imax + 1] = t0/psi2                                                            # Set initial temperature
    center_array[1, imax + 1] = M[0, imax + 1]*np.sqrt(gamma*R_air*center_array[2, imax + 1])      # Set initial velocity
    p[0, imax + 1]            = p0/psi2**const2                                                    # Set initial pressure
    center_array[0, imax + 1] = p[0, imax + 1]/(R_air*center_array[2, imax + 1])
    
    # if M[0, 0] < 0.11668889438289902/100:
    #     M[0, 0] = 0.11668889438289902/100
    # if center_array[1, 0] < 57.2706378650403/100:
    #     center_array[1, 0] = 57.2706378650403/100
    
    # if M[0, imax + 1] < 0.11668889438289902/100:
    #     M[0, imax + 1] = 0.11668889438289902/100
    # if center_array[1, imax + 1] < 57.2706378650403/100:
    #     center_array[1, imax + 1] = 57.2706378650403/100

#try_boundary_conditions()

# ---------- Construct conserved and flux vector ------------

# Compute source terms
S       = np.zeros((3, imax))
S[1, :] = p[0, cell_alias]*dA_dx
# -----------

U = np.zeros((3, imax + 2))

U[0, :] = center_array[0, :]                    # rho
U[1, :] = center_array[0, :]*center_array[1, :] # rho*u
U[2, :] = p[0, :]/(gamma - 1) + 0.5*center_array[0, :]*center_array[1, :]**2

def compute_primitive_variables():
    center_array[0, cell_alias] = U[0, cell_alias]
    center_array[1, cell_alias] = U[1, cell_alias]/center_array[0, cell_alias]
    p[0, cell_alias] = p0*(center_array[0, cell_alias]/rho0)**gamma
    M[0, cell_alias] = center_array[1, cell_alias]/np.sqrt(gamma*p[0, cell_alias]/center_array[0, cell_alias])
    center_array[2, :] = t0/(1 + const1*M[0, :]**2)

F = np.zeros((3, imax + 1))

def compute_fluxes():
    for i in range(imax + 1):
        F[0, i] = (center_array[0, i]*center_array[1, i] + center_array[0, i + 1]*center_array[1, i + 1])/2
        F[1, i] = (center_array[0, i]*center_array[1, i]**2 + p[0, i] + center_array[0, i + 1]*center_array[1, i + 1]**2 + p[0, i + 1])/2
        F[2, i] = (const2*p[0, i]*center_array[1, i] + 0.5*center_array[0, i]*center_array[1, i]**3 + const2*p[0, i + 1]*center_array[1, i + 1] + 0.5*center_array[0, i + 1]*center_array[1, i + 1]**3)
        # F[0, i] = center_array[0, i]*center_array[1, i]
        # F[1, i] = center_array[0, i]*center_array[1, i]**2 + p[0, i]
        # F[2, i] = const2*p[0, i]*center_array[1, i] + 0.5*center_array[0, i]*center_array[1, i]**3
compute_fluxes()

D              = np.zeros((3, imax + 1))
D2             = np.zeros((3, imax + 1))
D4             = np.zeros((3, imax + 1))
lambda_max     = np.zeros((1, imax + 2))
nu             = np.zeros((1, imax + 2))
p_extrapolated = np.zeros((1, imax + 4))
epsilon2       = np.zeros((1, imax + 1))
epsilon4       = np.zeros((1, imax + 1))
lambda_max = center_array[1, :] + center_array[1, :]/M[0, :]
M_extrapolated = np.zeros((1, imax + 4))

def compute_dissipation():
    M_extrapolated[0, 1:imax + 3] = M[0, :]
    M_extrapolated[0, 0]          = 2*M_extrapolated[0, 1] - M_extrapolated[0, 2]
    M_extrapolated[0, imax + 3]   = 2*M_extrapolated[0, imax + 2]  - M_extrapolated[0, imax + 1]
    
    p_extrapolated = p0/(1 + const1*M_extrapolated[0, :]**2)**const2
    
    # U[:, 0] =2*U[:, 1] - U[:, 2]
    # U[:, imax + 1] =2*U[:, imax] - U[:, imax - 1]
    
    # U[0, 0] = center_array[0, 0]                    # rho
    # U[1, 0] = center_array[0, 0]*center_array[1, 0] # rho*u
    # U[2, 0] = p[0, 0]/(gamma - 1) + 0.5*center_array[0, 0]*center_array[1, 0]**2
    
    # U[0, imax + 1] = center_array[0, imax + 1]                    # rho
    # U[1, imax + 1] = center_array[0, imax + 1]*center_array[1, imax + 1] # rho*u
    # U[2, imax + 1] = p[0, imax + 1]/(gamma - 1) + 0.5*center_array[0, imax + 1]*center_array[1, imax + 1]**2
    
    # p_extrapolated[0, 1:imax + 3] = p[0, :]
    # p_extrapolated[0, 0]          = 2*p_extrapolated[0, 1] - p_extrapolated[0, 2]
    # p_extrapolated[0, imax + 3]   = 2*p_extrapolated[0, imax + 2]  - p_extrapolated[0, imax + 1]
    
    for i in range(imax + 2):
        # nu[0, i] = abs((p_extrapolated[0, i + 2] - 2*p_extrapolated[0, i + 1] + p[0, i])/(p_extrapolated[0, i + 2] + 2*p_extrapolated[0, i + 1] + p_extrapolated[0, i]))
        nu[0, i] = abs((p_extrapolated[i + 2] - 2*p_extrapolated[i + 1] + p[0, i])/(p_extrapolated[i + 2] + 2*p_extrapolated[i + 1] + p_extrapolated[i]))
    
    epsilon2[0, 0]    = kappa2*max(nu[0, 0], nu[0, 1], nu[0, 2])
    epsilon2[0, imax] = kappa2*max(nu[0, imax - 1], nu[0, imax], nu[0, imax + 1])
    
    for i in range(1, imax):
        epsilon2[0, i] = kappa2*max(nu[0, i - 1], nu[0, i], nu[0, i + 1], nu[0, i + 2])
        
    epsilon4[0, :] = np.maximum((kappa4 - epsilon2), np.zeros((1, imax + 1)))
    
    for i in range(imax + 1):
        D2[:, i] = ((lambda_max[i] + lambda_max[i + 1])/2)*epsilon2[0, i]*(U[:, i + 1] - U[:, i])
    
    for i in range(1, imax - 1):
        D4[:, i] = ((lambda_max[i] + lambda_max[i + 1])/2)*epsilon4[0, i]*(U[:, i + 2] - 3*U[:, i + 1] + 3*U[:, i] - U[:, i - 1])
    
    D4[:, 0]        = 2*D4[:, 1] - D4[:, 1]
    D4[:, imax - 1] = 2*D4[:, imax - 2] - D4[:, imax - 3]
    D4[:, imax]     = 2*D4[:, imax - 1] - D4[:, imax - 2]

D = -(D2 - D4)

# TRY NO U GHOST CELLS



# ---------- Calculate residual (L2) ----------

init_norm = np.zeros((3, 1))
R_i = np.zeros((3, imax))

for i in range(imax):
    R_i[:, i] = (F[:, i + 1] + D[:, i + 1])*A_intf[i + 1] - (F[:, i] + D[:, i])*A_intf[i] - S[:, i]*dx

init_norm[0] = ((np.sum(R_i[0, :]**2))/imax)**0.5 # continuity
init_norm[1] = ((np.sum(R_i[1, :]**2))/imax)**0.5 # x-momentum
init_norm[2] = ((np.sum(R_i[2, :]**2))/imax)**0.5 # energy

norm = np.zeros((3, 1))
res = np.zeros((3, 1))

def check_iterative_convergence():
    norm[0] = ((np.sum(R_i[0, :]**2))/imax)**0.5 # continuity
    norm[1] = ((np.sum(R_i[1, :]**2))/imax)**0.5 # x-momentum
    norm[2] = ((np.sum(R_i[2, :]**2))/imax)**0.5 # energy
    
    res[0] = norm[0]/init_norm[0]
    res[1] = norm[1]/init_norm[1]
    res[2] = norm[2]/init_norm[2]
    
    if j%100 == 0:
        f1.write(str(j) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])) + '\n')
    
    print(str(j) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])))        

# ---------- Compute DE L2 norm ----------

def de_norms():
    DE = np.zeros((3, imax))
    DE_norm = np.zeros((3, 1))
    
    DE[0, :] = exact_solution[5, :] -  center_array[0, cell_alias]
    DE[1, :] = exact_solution[6, :] -  center_array[1, cell_alias]
    DE[2, :] = exact_solution[3, :] -  center_array[2, cell_alias]
    
    DE_norm[0] = ((np.sum(DE[0, :]**2))/imax)**0.5
    DE_norm[1] = ((np.sum(DE[1, :]**2))/imax)**0.5
    DE_norm[2] = ((np.sum(DE[2, :]**2))/imax)**0.5

    f3 = open('DE_norms.txt', 'a')
    print('DE Norms:\n' + 'rho = ' + str(float(DE_norm[0])) + ' ' + 'u = ' + str(float(DE_norm[1])) + ' ' + 'T = ' + str(float(DE_norm[2])))
    f3.write('DE Norms:\n' + 'rho = ' + str(float(DE_norm[0])) + ' ' + 'u = ' + str(float(DE_norm[1])) + ' ' + 'T = ' + str(float(DE_norm[2])))
    f3.close()
# ---------- Write to file ----------

def write_to_file():
    f.write('zone T="' + str(j) + '"\n')
    f.write('I=' + str(imax) + '\n')
    f.write('DATAPACKING=POINT\n')
    f.write('DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE)\n')
    for g in range(imax):
        f.write(str(float(x_cell[g]))  + " " +  str(float(A_cell[g]))  + " " +  str(float(M[0, g]))  + " " +  str(float(center_array[0, g]))  + " " +  str(float(center_array[1, g]))  + " " +  str(float(center_array[2, g]))  + " " +  str(float(p[0, g])) + '\n')

# ---------- Compute timestep (local timestepping) ----------

dt = np.zeros((1, imax))
dt = cfl*dx/lambda_max[cell_alias]

# ---------- Main Loop ----------

print('Iteration' + " " + 'Continuity' + " " + 'X - mtm' + " " + 'Energy')

f = open('soln.dat', 'a')
f.write('TITLE = "Quasi-1D Nozzle Solution"\n')
f.write('variables="x(m)""Area(m^2)""Mach""rho(kg/m^3)""u(m/s)""T(K)""Press(N/m^2)"\n')

f1 = open('res.dat', 'a')
f1.write('TITLE = "Quasi-1D Nozzle Residuals"\n')
f1.write('variables="Iteration""Coninuity""X-momentum""Energy"\n')

for j in range(nmax + 1):
    lambda_max = center_array[1, :] + center_array[1, :]/M[0, :]
    dt = cfl*dx/lambda_max[cell_alias]
    compute_fluxes()
    compute_dissipation()
    D = -(D2 - D4)
    S[1, :] = p[0, cell_alias]*dA_dx
    for i in range(imax):
        R_i[:, i] = (F[:, i + 1] + D[:, i + 1])*A_intf[i + 1] - (F[:, i] + D[:, i])*A_intf[i] - S[:, i]*dx
    U[:, cell_alias] = U[:, cell_alias] - (dt/V)*R_i
    compute_primitive_variables()
    #try_boundary_conditions()
    set_boundary_conditions()
    check_iterative_convergence()
    if j%2000 == 0:
        write_to_file()
    if j == 20000:
        cfl = 0.05
    if j == 30000:
        cfl = 0.1
    if res[0] <= 1E-10 and res[1] <= 1E-10 and res[2] <= 1E-10 :
        print('Solution converged in ' + str(j) + ' iterations')
        break

f.close()
f1.close()
if shock_flag == 0:
    de_norms()
