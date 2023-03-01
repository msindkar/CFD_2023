def supersonic_nozzle(p0, t0, imax):

    import numpy as np
    
    x = np.arange(-1 + 1/imax, 1 + 1/imax, 2/imax)
    # p0 = 3E5    # in kPa
    # t0 = 600
    gamma = 1.4
    R = 8314    # J/(kmol*K)
    m_air = 28.96 # 
    R_air = R/m_air # Specific gas constant for air
    
    A = 0.2 + 0.4*(1 + np.sin(np.pi*(x - 0.5)))
    A_star = 0.2 + 0.4*(1 + np.sin(np.pi*(-0.5)))
    A_local = np.array([A[0], A[1], A[3], A[4]])
    A_bar = A_local/A_star
    
    M_new = np.array([0.8, 0.9, 8, 10]) # Mach number initial guess for newton solver
    M_old, F_M, dF_dM, phi = np.zeros(4), np.ones(4), np.zeros(4), np.zeros(4)
    
    const1 = 2/(gamma + 1)
    const2 = (gamma - 1)/2
    const3 = (gamma + 1)/(gamma - 1)
    const4 = 2/(gamma - 1)
    const5 = gamma/(gamma - 1)
    
    while np.abs(F_M[0]) >= 1E-14 or np.abs(F_M[1]) >= 1E-14 or np.abs(F_M[2]) >= 1E-13 or np.abs(F_M[3]) >= 2*1E-13:
        M_old = M_new
        phi = const1 * (1 + const2*(M_old**2))
        F_M = phi**const3 - (A_bar**2)*(M_old**2)
        dF_dM = 2*M_old*(phi**const4 - A_bar**2)
        M_new = M_old - F_M/dF_dM
    
    solution_vector = np.zeros((7,5))
    solution_vector[0, :] = np.array((x[0], x[1], 0, x[2], x[3])) #x-position
    solution_vector[1, :] = A #area
    solution_vector[2, :] = np.array([M_new[0], M_new[1], 1, M_new[2], M_new[3]]) #Mach number
    
    psi = 1 + const2*solution_vector[2, :]**2
    
    solution_vector[3,:] = t0/psi #temperature
    solution_vector[4,:] = p0/psi**const5 #pressure
    solution_vector[5,:] = solution_vector[4,:]/(R_air*solution_vector[3,:]) #density
    solution_vector[6,:] = solution_vector[2, :]*(gamma*R_air*solution_vector[3,:])**0.5 #velocity
    
    return solution_vector