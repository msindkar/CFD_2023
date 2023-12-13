def src_mms_corrected(imax, jmax, imaxc, jmaxc, x, y, x_cell, y_cell):
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
    
    sin   = np.sin
    cos   = np.cos
    Pi    = np.pi
    gamma = 1.4
    
    VE = np.zeros((imaxc, jmaxc, 4))
    SE = np.zeros((imaxc, jmaxc, 4))
    
    for i in np.arange(imaxc):
        for j in np.arange(jmaxc):
            
                x = x_cell[i][j]
                y = y_cell[i][j]
                
                VE[i][j][0] = rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L);
                VE[i][j][1] = uvel0 + uvely*cos((3.*Pi*y)/(5.*L)) + uvelx*sin((3.*Pi*x)/(2.*L));
                VE[i][j][2] = vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2.*Pi*y)/(3.*L));
                VE[i][j][3] = press0 + pressx*cos((2.*Pi*x)/L) + pressy*sin((Pi*y)/L);
    
                SE[i][j][0] = (3*Pi*uvelx*cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L)))/(2.*L) + \
                            (2*Pi*vvely*cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L)))/(3.*L) + \
                            (Pi*rhox*cos((Pi*x)/L)*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L))))/L - \
                            (Pi*rhoy*sin((Pi*y)/(2.*L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L))))/(2.*L)
                            
                SE[i][j][1] = (3*Pi*uvelx*cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*\
                            (uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L))))/L + (2*Pi*vvely*cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + \
                            rhox*sin((Pi*x)/L))*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L))))/(3.*L) + (Pi*rhox*cos((Pi*x)/L)*pow(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + \
                            uvelx*sin((3*Pi*x)/(2.*L)),2.)/L - 2*Pi*pressx*sin((2*Pi*x)/L))/L - (Pi*rhoy*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + \
                            uvelx*sin((3*Pi*x)/(2.*L)))*sin((Pi*y)/(2.*L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L))))/(2.*L) - \
                            (3*Pi*uvely*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*sin((3*Pi*y)/(5.*L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + \
                            vvely*sin((2*Pi*y)/(3.*L))))/(5.*L)
                            
                SE[i][j][2] = (Pi*pressy*cos((Pi*y)/L))/L - (Pi*vvelx*sin((Pi*x)/(2.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + \
                            uvelx*sin((3*Pi*x)/(2.*L))))/(2.*L) + (3*Pi*uvelx*cos((3*Pi*x)/(2.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + \
                            vvely*sin((2*Pi*y)/(3.*L))))/(2.*L) + (4*Pi*vvely*cos((2*Pi*y)/(3.*L))*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + \
                            vvely*sin((2*Pi*y)/(3.*L))))/(3.*L) + (Pi*rhox*cos((Pi*x)/L)*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + \
                            vvely*sin((2*Pi*y)/(3.*L))))/L - Pi*rhoy*sin((Pi*y)/(2.*L))*pow(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)),2.0)/(2.*L)\
                                                                                                                                                            
                SE[i][j][3] = (uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)))*\
                            ((-2*Pi*pressx*sin((2*Pi*x)/L))/L + (rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*\
                            ((-2*Pi*pressx*sin((2*Pi*x)/L))/((-1 + gamma)*L*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))) + \
                            ((3*Pi*uvelx*cos((3*Pi*x)/(2.*L))*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L))))/L - \
                            (Pi*vvelx*sin((Pi*x)/(2.*L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L))))/L)/2. - \
                            (Pi*rhox*cos((Pi*x)/L)*(press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L)))/ \
                            ((-1 + gamma)*L*pow(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L),2.))) + \
                            (Pi*rhox*cos((Pi*x)/L)*((pow(wvel0,2.) + pow(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)),2.) + \
                            pow(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)),2.))/2. + \
                            (press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L))/ \
                            ((-1 + gamma)*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L)))))/L) + \
                            (3*Pi*uvelx*cos((3*Pi*x)/(2.*L))*(press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L) + \
                            (rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*\
                            ((pow(wvel0,2.) + pow(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)),2.) + \
                            pow(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)),2.))/2. + \
                            (press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L))/ \
                            ((-1 + gamma)*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))))))/(2.*L) + \
                            (2*Pi*vvely*cos((2*Pi*y)/(3.*L))*(press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L) + \
                            (rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))*\
                            ((pow(wvel0,2.) + pow(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)),2.) + \
                            pow(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)),2))/2. + \
                            (press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L))/ \
                            ((-1 + gamma)*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))))))/(3.*L) + \
                            (vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)))* \
                            ((Pi*pressy*cos((Pi*y)/L))/L - (Pi*rhoy*sin((Pi*y)/(2.*L))* \
                            ((pow(wvel0,2.) + pow(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)),2.) + \
                            pow(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L)),2.))/2. + \
                            (press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L))/ \
                            ((-1 + gamma)*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L)))))/(2.*L) + \
                            (rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))* \
                            ((Pi*pressy*cos((Pi*y)/L))/((-1 + gamma)*L*(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L))) + \
                            ((-6*Pi*uvely*(uvel0 + uvely*cos((3*Pi*y)/(5.*L)) + uvelx*sin((3*Pi*x)/(2.*L)))*sin((3*Pi*y)/(5.*L)))/(5.*L) + \
                            (4*Pi*vvely*cos((2*Pi*y)/(3.*L))*(vvel0 + vvelx*cos((Pi*x)/(2.*L)) + vvely*sin((2*Pi*y)/(3.*L))))/(3.*L))/2. + \
                            (Pi*rhoy*sin((Pi*y)/(2.*L))*(press0 + pressx*cos((2*Pi*x)/L) + pressy*sin((Pi*y)/L)))/ \
                            (2.*(-1 + gamma)*L*pow(rho0 + rhoy*cos((Pi*y)/(2.*L)) + rhox*sin((Pi*x)/L),2.)))) 
    
    return VE, SE