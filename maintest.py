import time  as t
import numpy as np

from mesh_loader  import mesh_loader
from BC_INIT_TEST import load_BC_init
from area_calc    import area_normals_calc
from src_mms      import src_mms
from config       import flux_scheme

R = 8314        # J/(kmol*K)
m_air = 28.96   # 
R_air = R/m_air # Specific gas constant for air
gamma = 1.4
W = 1

nmax  = 10000   # max iterations
n = 0         # current iteration (REMOVE WHEN DONE)
cfl = 0.8
conv= 0

nan_flag = 0

imax, jmax, x, y, x_cell, y_cell = mesh_loader()
prim1, T1, BC_left, BC_bot, BC_right, BC_up, BC_left_T, BC_bot_T, BC_right_T, BC_up_T = load_BC_init(R_air, imax, jmax, x_cell, y_cell)
A_vert, A_hori, normals_h, normals_v = area_normals_calc(imax, jmax, x, y)
src_array = src_mms(R_air, imax, jmax, x_cell, y_cell)

imaxc, jmaxc = imax - 1, jmax - 1

# ---------- 3D Area array to allow element by element multiplication ----------
A_vert4 = np.zeros((imax, jmax - 1, 4))
A_hori4 = np.zeros((imax - 1, jmax, 4))

for k in np.arange(4):
    A_vert4[:, :, k] = A_vert[:, :]
    A_hori4[:, :, k] = A_hori[:, :]
#---------- Calculating cell dx, dy by averaging node dx, dy ----------
dx = np.zeros((imaxc, jmaxc))
dy = np.zeros((imaxc, jmaxc))

dx1 = np.zeros((imaxc, jmax))
dy1 = np.zeros((imax, jmaxc))

for i in np.arange(imaxc):
    dx1[i, :] = x[i + 1, :] - x[i, :]

for j in np.arange(jmaxc):
    dx[:, j] = (dx1[:, j + 1] + dx1[:, j])*0.5
    
for j in np.arange(jmaxc):
    dy1[:, j] = y[:, j + 1] - y[:, j]

for i in np.arange(imaxc):
    dy[i, :]= (dy1[i + 1, :] + dy1[i, :])*0.5
    
del dx1, dy1
#---------- Calculating cell 'volumes' ----------
V = np.zeros((imaxc, jmaxc))

for j in np.arange(jmaxc):
    for i in np.arange(imaxc):
        AC = np.sqrt((x[i + 1, j + 1] - x[i, j])**2 + (y[i + 1, j + 1] - y[i, j])**2)
        BD = np.sqrt((x[i + 1, j] - x[i, j + 1])**2 + (y[i + 1, j] - y[i, j + 1])**2)
        V[i, j] = 0.5*W*(AC*BD)
#---------- Calculating cell centered areas ----------
A_v_cell = np.zeros((imaxc, jmaxc))
A_h_cell = np.zeros((imaxc, jmaxc))

for i in np.arange(imaxc):
    A_v_cell[i, :] = (A_vert[i + 1, :] + A_vert[i, :])*0.5
    
for j in np.arange(jmaxc):
    A_h_cell[:, j] = (A_hori[:, j + 1] + A_hori[:, j])*0.5
#---------- Assign to arrays with ghost cells ----------
prim = np.zeros((imaxc + 4, jmaxc + 4, 4))
prim[2:imaxc + 2, 2:jmaxc + 2, :] = prim1[:, :, :]
prim[:, :, 0][prim[:, :, 0] == 0.0] = 1

T    = np.zeros((imaxc + 4, jmaxc + 4))
T[2:imaxc + 2, 2:jmaxc + 2] = T1[:, :]

del T1, prim1
#---------- Calculate cell centered averaged normals ----------
normals_h_c = np.zeros((imaxc, jmaxc, 2))
normals_v_c = np.zeros((imaxc, jmaxc, 2))

for i in np.arange(imaxc):
    normals_v_c[i, :] = (normals_v[i + 1, :] + normals_v[i, :])*0.5
    
for j in np.arange(jmaxc):
    normals_h_c[:, j] = (normals_h[:, j + 1] + normals_h[:, j])*0.5
#--------------------
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
    a[:, :] = (gamma*R_air*prim[:, :, 3]/prim[:, :, 0])**(0.5)
    a[a[:, :] == 0] = 1
a_calc()

vel2 = np.zeros((imaxc + 4, jmaxc + 4))
def vel2_calc():
    vel2[:, :] = prim[:, :, 1]**2 + prim[:, :, 2]**2
vel2_calc()

ht   = np.zeros((imaxc + 4, jmaxc + 4))
def ht_calc():
    ht  [:, :] = (gamma*prim[:, :, 3])/((gamma - 1)*prim[:, :, 0]) + 0.5*(vel2[:, :])
ht_calc()

et = np.zeros((imaxc + 4, jmaxc + 4))
def et_calc():
    et[:, :] = prim[:, :, 3]/((gamma - 1)*prim[:, :, 0])+ 0.5*(vel2[:, :])
et_calc()

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

epsilon = 0 # flux order switch

F_v = np.zeros((imax, jmax - 1, 4))
F_h = np.zeros((imax - 1, jmax, 4))
F_v_plus = np.zeros((imax, jmax - 1, 4))
F_h_plus = np.zeros((imax - 1, jmax, 4))
F_v_minus= np.zeros((imax, jmax - 1, 4))
F_h_minus= np.zeros((imax - 1, jmax, 4))

def van_leer_flux(normals_v, normals_h): # arg is iteration number?
    # st = t.time()
    for i in np.arange(0, imax): # vertical interface indexing
        i_p = i + 1
        ucap_L_v = prim[i_p, alias_cj, 1]*normals_v[i, :, 0] + prim[i_p, alias_cj, 2]*normals_v[i, :, 1]
        ucap_R_v = prim[i_p + 1, alias_cj, 1]*normals_v[i, :, 0] + prim[i_p + 1, alias_cj, 2]*normals_v[i, :, 1]
        M_L = ucap_L_v/a[i_p, alias_cj]
        M_R = ucap_R_v/a[i_p + 1, alias_cj]
        M_plus = 0.25*(M_L + 1)**2
        M_minus= -0.25*(M_R + 1)**2
        beta_L = -np.maximum(np.zeros(np.size(M_L), dtype = int), (1 - M_L.astype(int)))
        beta_R = -np.maximum(np.zeros(np.size(M_R), dtype = int), (1 - M_R.astype(int)))
        alpha_plus = (1/2)*(1 + np.sign(M_L))
        alpha_minus= (1/2)*(1 - np.sign(M_R))
        c_plus = alpha_plus*(1 + beta_L)*M_L - beta_L*M_plus
        c_minus= alpha_minus*(1 + beta_R)*M_R - beta_R*M_minus
        F_C_p = prim[i_p, alias_cj, 0]*a[i_p, alias_cj]*c_plus[:]*np.array([np.ones(jmax - 1), prim[i_p, alias_cj, 1], prim[i_p, alias_cj, 2], ht[i_p, alias_cj]])
        F_C_m = prim[i_p + 1, alias_cj, 0]*a[i_p + 1, alias_cj]*c_minus[:]*np.array([np.ones(jmax - 1), prim[i_p + 1, alias_cj, 1], prim[i_p + 1, alias_cj, 2], ht[i_p + 1, alias_cj]])
        P_2bar_plus = M_plus*(- M_L + 2)
        P_2bar_minus = M_minus*(- M_R - 2)
        D_plus = alpha_plus*(1 + beta_L) - beta_L*P_2bar_plus
        D_minus = alpha_minus*(1 + beta_R) - beta_R*P_2bar_minus
        F_P_p = D_plus*np.array([np.zeros(jmax - 1), normals_v[i, :, 0]*prim[i_p, alias_cj, 3], normals_v[i, :, 1]*prim[i_p, alias_cj, 3], np.zeros(jmax - 1)])
        F_P_m = D_minus*np.array([np.zeros(jmax - 1), normals_v[i, :, 0]*prim[i_p + 1, alias_cj, 3], normals_v[i, :, 1]*prim[i_p + 1, alias_cj, 3], np.zeros(jmax - 1)])
        F_C_p = F_C_p.transpose()
        F_C_m = F_C_m.transpose()
        F_P_p = F_P_p.transpose()
        F_P_m = F_P_m.transpose()
        F_v[i, :, :] = F_C_p + F_P_p + F_C_m + F_P_m
        
    for j in np.arange(0, jmax):
        j_p = j + 1
        ucap_L_h = prim[alias_ci, j_p, 1]*normals_h[:, j, 0] + prim[alias_ci, j_p, 2]*normals_h[:, j, 0]
        ucap_R_h = prim[alias_ci, j_p + 1, 1]*normals_h[:, j, 0] + prim[alias_ci, j_p + 1, 2]*normals_h[:, j, 0]
        M_L = ucap_L_h/a[alias_ci, j_p]
        M_R = ucap_R_h/a[alias_ci, j_p + 1]
        M_plus = 0.25*(M_L + 1)**2
        M_minus= -0.25*(M_R + 1)**2
        beta_L = -np.maximum(np.zeros(np.size(M_L), dtype = int), (1 - M_L.astype(int)))
        beta_R = -np.maximum(np.zeros(np.size(M_R), dtype = int), (1 - M_R.astype(int)))
        alpha_plus = (1/2)*(1 + np.sign(M_L))
        alpha_minus= (1/2)*(1 - np.sign(M_R))
        c_plus = alpha_plus*(1 + beta_L)*M_L - beta_L*M_plus
        c_minus= alpha_minus*(1 + beta_R)*M_R - beta_R*M_minus
        F_C_p = prim[alias_ci, j_p, 0]*a[alias_ci, j_p]*c_plus[:]*np.array([np.ones(imax - 1), prim[alias_ci, j_p, 1], prim[alias_ci, j_p, 2], ht[alias_ci, j_p]])
        F_C_m = prim[alias_ci, j_p + 1, 0]*a[alias_ci, j_p + 1]*c_minus[:]*np.array([np.ones(imax - 1), prim[alias_ci, j_p + 1, 1], prim[alias_ci, j_p + 1, 2], ht[alias_ci, j_p + 1]])
        P_2bar_plus = M_plus*(- M_L + 2)
        P_2bar_minus = M_minus*(- M_R - 2)
        D_plus = alpha_plus*(1 + beta_L) - beta_L*P_2bar_plus
        D_minus = alpha_minus*(1 + beta_R) - beta_R*P_2bar_minus
        F_P_p = D_plus*np.array([np.zeros(imax - 1), normals_h[:, j, 0]*prim[alias_ci, j_p, 3], normals_h[:, j, 0]*prim[alias_ci, j_p, 3], np.zeros(imax - 1)])
        F_P_m = D_minus*np.array([np.zeros(imax - 1), normals_h[:, j, 0]*prim[alias_ci, j_p + 1, 3], normals_h[:, j, 0]*prim[alias_ci, j_p + 1, 3], np.zeros(imax - 1)])
        F_C_p = F_C_p.transpose()
        F_C_m = F_C_m.transpose()
        F_P_p = F_P_p.transpose()
        F_P_m = F_P_m.transpose()
        F_h[:, j, :] = F_C_p + F_P_p + F_C_m + F_P_m
    # et = t.time()
    # print('Flux calculated in: ' + str(et - st) + 's')
    return F_h[:, :, :], F_v[:, :, :]

if flux_scheme == 'vl1':
    selected_flux = van_leer_flux
else:
    print('Invalid value for flux_scheme selection, please check config.py')
    raise SystemExit(0)

F_h_plus[:, :, :], F_v_plus[:, :, :] = selected_flux(normals_v, normals_h)
F_h_minus[:, :, :], F_v_minus[:, :, :] = selected_flux(-normals_v, -normals_h) # USE THIS TO CALC FLUXES

cons = np.zeros((imaxc, jmaxc,  4)) # DON'T NEED CONS GHOST CELLS

def primitive_to_conserved_variables():
    cons[:, :, 0] = prim[2:imaxc + 2, 2:jmaxc + 2, 0]
    cons[:, :, 1] = prim[2:imaxc + 2, 2:jmaxc + 2, 0]*prim[2:imaxc + 2, 2:jmaxc + 2, 1]
    cons[:, :, 2] = prim[2:imaxc + 2, 2:jmaxc + 2, 0]*prim[2:imaxc + 2, 2:jmaxc + 2, 2]
    cons[:, :, 3] = prim[2:imaxc + 2, 2:jmaxc + 2, 0]*et[2:imaxc + 2, 2:jmaxc + 2]
primitive_to_conserved_variables()

def conserved_to_primitive_variables():
    prim[2:imaxc + 2, 2:jmaxc + 2, 0] = cons[:, :, 0]
    prim[2:imaxc + 2, 2:jmaxc + 2, 1] = cons[:, :, 1]/cons[:, :, 0]
    prim[2:imaxc + 2, 2:jmaxc + 2, 2] = cons[:, :, 2]/cons[:, :, 0]
    prim[2:imaxc + 2, 2:jmaxc + 2, 3] = cons[:, :, 3]*(0.4) - 0.5*(0.4)*(prim[2:imaxc + 2, 2:jmaxc + 2, 1]**2 + prim[2:imaxc + 2, 2:jmaxc + 2, 2]**2)

init_norm = np.zeros((4, 1))
R_ij = np.zeros((imaxc, jmaxc, 4))
norm = np.zeros((4, 1))
res = np.zeros((4, 1))
    
def calculate_residuals(): # Cell centered indexing
    for i in np.arange(0, imaxc):
        R_ij[i, :, :] = F_v_plus[i + 1, :, :]*A_vert4[i + 1, :, :] - F_v_minus[i, :, :]*A_vert4[i, :, :] - src_array[i, :, :]
    for j in np.arange(0, jmaxc):
        R_ij[:, j, :] = R_ij[:, j, :] + F_h_plus[:, j + 1, :]*A_hori4[:, j + 1, :] - F_h_minus[:, j, :]*A_hori4[:, j, :] #- src_array[:, j, :]
calculate_residuals()

init_norm[0] = ((np.sum(R_ij[:, :, 0]**2))/(imaxc*jmaxc))**0.5 # continuity
init_norm[1] = ((np.sum(R_ij[:, :, 1]**2))/(imaxc*jmaxc))**0.5 # x-momentum
init_norm[2] = ((np.sum(R_ij[:, :, 2]**2))/(imaxc*jmaxc))**0.5 # y-momentum
init_norm[3] = ((np.sum(R_ij[:, :, 3]**2))/(imaxc*jmaxc))**0.5 # energy

f1 = open('res.dat', 'w')
f1.write('TITLE = "2D Euler Solver Residuals"\n')
f1.write('variables="Iteration""Coninuity""X-momentum""Y-momentum""Energy"\n')

def out_steady_state_iterative_residuals():
    norm[0] = ((np.sum(R_ij[:, :, 0]**2))/(imaxc*jmaxc))**0.5 # continuity
    norm[1] = ((np.sum(R_ij[:, :, 1]**2))/(imaxc*jmaxc))**0.5 # x-momentum
    norm[2] = ((np.sum(R_ij[:, :, 2]**2))/(imaxc*jmaxc))**0.5 # y-momentum
    norm[3] = ((np.sum(R_ij[:, :, 3]**2))/(imaxc*jmaxc))**0.5 # energy
    
    res[0] = norm[0]/init_norm[0]
    res[1] = norm[1]/init_norm[1]
    res[2] = norm[2]/init_norm[2]
    res[3] = norm[3]/init_norm[3]
    
    if n%100 == 0:
        f1.write(str(n) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])) + " " + str(float(res[3])) +'\n')
        print(str(n) + " " + str(float(res[0])) + " " + str(float(res[1])) + " " + str(float(res[2])) + " " + str(float(res[3])))

lam_h = np.zeros((imaxc, jmaxc))
lam_v = np.zeros((imaxc, jmaxc))

dt = np.zeros((imaxc, jmaxc))
dt4 = np.zeros((imaxc, jmaxc, 4))

def compute_time_step():
    lam_v[:, :] = np.abs(prim[2:imaxc + 2, 2:jmaxc + 2, 1]*normals_v_c[:, :, 0] + prim[2:imaxc + 2, 2:jmaxc + 2, 2]*normals_v_c[:, :, 1]) + a[2:imaxc + 2, 2:jmaxc + 2]
    lam_h[:, :] = np.abs(prim[2:imaxc + 2, 2:jmaxc + 2, 1]*normals_h_c[:, :, 0] + prim[2:imaxc + 2, 2:jmaxc + 2, 2]*normals_v_c[:, :, 1]) + a[2:imaxc + 2, 2:jmaxc + 2]
    
    dt[:, :] = cfl*(V)/(lam_v[:, :]*A_v_cell[:, :] + lam_h[:, :]*A_h_cell[:, :])
    for k in np.arange(4):
        dt4[:, :, k] = dt[:, :]#/V[:, :]
    
compute_time_step()

for n in range(nmax + 1):
    F_h_plus[:, :, :], F_v_plus[:, :, :] = selected_flux(normals_v, normals_h)
    F_h_minus[:, :, :], F_v_minus[:, :, :] = selected_flux(-normals_v, -normals_h)
    compute_time_step()
    # source_terms()
    primitive_to_conserved_variables()
    calculate_residuals()
    cons = cons - R_ij*(dt4)
    conserved_to_primitive_variables()
    a_calc()
    vel2_calc()
    ht_calc()
    et_calc()
    out_steady_state_iterative_residuals()
    if (res[:, :] <= 1E-8).all():
        print('Solution converged in ' + str(n) + ' iterations')
        # write_to_file()
        break

f1.close()