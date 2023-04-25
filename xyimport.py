import numpy as np
rho0 = 0.042808
ue   = 872.359
u = np.loadtxt('u.xy')
rho = np.loadtxt('rho.xy')
d2_cell_wise = (rho*u/(rho0*ue))*(1 - u/ue)
d2 = np.sum(d2_cell_wise, 1)
d1_cell_wise = (1 - rho*u/(rho0*ue))
d1 = np.sum(d1_cell_wise, 1)