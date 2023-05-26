def load_BC_init(imax, jmax, x_cell, y_cell):    
    # BC/INIT CALC
    from config import supersonic
    import numpy as np
    
    L = 1
    
    if supersonic == 1:
        phi_rho = np.array([1.0, 0.15, -0.1, 1.0, 0.5]) # phi0 phix phiy aphix aphiy
        phi_u   = np.array([800.0, 50.0, -30.0, 1.5, 0.6])
        phi_v   = np.array([800.0, -75.0, 40.0, 0.5, 2/3])
        phi_p   = np.array([1.0e5, 0.2e5, 0.5e5, 2, 1])
    elif supersonic == 0:
        phi_rho = np.array([1.0, 0.15, -0.1, 1.0, 0.5]) # phi0 phix phiy aphix aphiy
        phi_u   = np.array([70.0, 5.0, -7.0, 1.5, 0.6])
        phi_v   = np.array([90.0, -15.0, 8.5, 0.5, 2/3])
        phi_p   = np.array([1.0e5, 0.2e5, 0.5e5, 2, 1])
    else:
        print('E: Invalid value in supersonic flag, please check the config.py file')
        raise SystemExit(0)
    
    prim = np.zeros((imax - 1, jmax - 1, 4)) # rho u v p
    
    prim[:, :, 0] = phi_rho[0] + phi_rho[1]*np.sin(phi_rho[3]*np.pi*x_cell/L) + phi_rho[2]*np.cos(phi_rho[4]*np.pi*y_cell/L)
    prim[:, :, 1] = phi_u[0]   + phi_u[1]*np.sin(phi_u[3]*np.pi*x_cell/L)     + phi_u[2]*np.cos(phi_u[4]*np.pi*y_cell/L)
    prim[:, :, 2] = phi_v[0]   + phi_v[1]*np.cos(phi_v[3]*np.pi*x_cell/L)     + phi_v[2]*np.sin(phi_v[4]*np.pi*y_cell/L)
    prim[:, :, 3] = phi_p[0]   + phi_p[1]*np.cos(phi_p[3]*np.pi*x_cell/L)     + phi_p[2]*np.sin(phi_p[4]*np.pi*y_cell/L)
    
    bot   = prim[:, 0, :]
    left  = prim[0, :, :]
    up    = prim[:, jmax - 2, :]
    right = prim[imax - 2, :, :]
    
    return prim, left, bot, right, up