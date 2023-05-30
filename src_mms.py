def src_mms(R_air, imax, jmax, x_cell, y_cell):
    import numpy as np
    from config import supersonic
    
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
    
    L = 1
        
    rho0 = phi_rho[0]
    rhox = phi_rho[1]
    rhoy = phi_rho[2]
    
    uvel0 = phi_u[0]
    uvelx = phi_u[1]
    uvely = phi_u[2]
    
    vvel0 = phi_v[0]
    vvelx = phi_v[1]
    vvely = phi_v[2]
    
    press0 = phi_p[0]
    pressx = phi_p[1]
    pressy = phi_p[2]
    
    wvel0 = 0
    
    Sin   = np.sin
    Cos   = np.cos
    Pi    = np.pi
    gamma = 1.4
    
    x = x_cell
    y = y_cell  # CAREFUL???????
    
    src_array = np.zeros((imax - 1, jmax - 1, 4))
    
    src_array[:, :, 0] = (3*Pi*uvelx*Cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L)))/(2.*L) + (2*Pi*vvely*Cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L)))/(3.*L) + (Pi*rhox*Cos((Pi*x)/L)*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L))))/L - (Pi*rhoy*Sin((Pi*y)/(2.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(2.*L)
    src_array[:, :, 1] = (3*Pi*uvelx*Cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L))))/L + (2*Pi*vvely*Cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L))))/(3.*L) + (Pi*rhox*Cos((Pi*x)/L)*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))**2)/L - (2*Pi*pressx*Sin((2*Pi*x)/L))/L - (Pi*rhoy*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))*Sin((Pi*y)/(2.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(2.*L) - (3*Pi*uvely*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*Sin((3*Pi*y)/(5.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(5.*L)
    src_array[:, :, 2] = (Pi*pressy*Cos((Pi*y)/L))/L - (Pi*vvelx*Sin((Pi*x)/(2.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L))))/(2.*L) + (3*Pi*uvelx*Cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(2.*L) + (4*Pi*vvely*Cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(3.*L) + (Pi*rhox*Cos((Pi*x)/L)*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/L - (Pi*rhoy*Sin((Pi*y)/(2.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))**2)/(2.*L)
    src_array[:, :, 3] = (uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))*((-2*Pi*pressx*Sin((2*Pi*x)/L))/L + (rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*((-2*Pi*pressx*Sin((2*Pi*x)/L))/((-1 + gamma)*L*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))) + ((3*Pi*uvelx*Cos((3*Pi*x)/(2.*L))*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L))))/L - (Pi*vvelx*Sin((Pi*x)/(2.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/L)/2. - (Pi*rhox*Cos((Pi*x)/L)*(press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L)))/((-1 + gamma)*L*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))**2)) + (Pi*rhox*Cos((Pi*x)/L)*((wvel0**2 + (uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))**2 + (vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))**2)/2. + (press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L))/((-1 + gamma)*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L)))))/L) + (3*Pi*uvelx*Cos((3*Pi*x)/(2.*L))*(press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L) + (rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*((wvel0**2 + (uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))**2 + (vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))**2)/2. + (press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L))/((-1 + gamma)*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))))))/(2.*L) + (2*Pi*vvely*Cos((2*Pi*y)/(3.*L))*(press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L) + (rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*((wvel0**2 + (uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))**2 + (vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))**2)/2. + (press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L))/((-1 + gamma)*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))))))/(3.*L) + (vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))*((Pi*pressy*Cos((Pi*y)/L))/L - (Pi*rhoy*Sin((Pi*y)/(2.*L))*((wvel0**2 + (uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))**2 + (vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L)))**2)/2. + (press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L))/((-1 + gamma)*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L)))))/(2.*L) + (rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))*((Pi*pressy*Cos((Pi*y)/L))/((-1 + gamma)*L*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))) + ((-6*Pi*uvely*(uvel0 + uvely*Cos((3*Pi*y)/(5.*L)) + uvelx*Sin((3*Pi*x)/(2.*L)))*Sin((3*Pi*y)/(5.*L)))/(5.*L) + (4*Pi*vvely*Cos((2*Pi*y)/(3.*L))*(vvel0 + vvelx*Cos((Pi*x)/(2.*L)) + vvely*Sin((2*Pi*y)/(3.*L))))/(3.*L))/2. + (Pi*rhoy*Sin((Pi*y)/(2.*L))*(press0 + pressx*Cos((2*Pi*x)/L) + pressy*Sin((Pi*y)/L)))/(2.*(-1 + gamma)*L*(rho0 + rhoy*Cos((Pi*y)/(2.*L)) + rhox*Sin((Pi*x)/L))**2)))
    
    return src_array