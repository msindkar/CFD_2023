def supersonic_nozzle_exact_solution(p0, t0, imax):

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
    # A_local = np.array([A[0], A[1], A[3], A[4]])
    A_bar = A/A_star
    
    M_new = x*1.4 + 1.6 # Mach number initial guess for newton solver
    for i in range(int(imax/2)):
        if M_new[i] >= 0.8:
            M_new[i] = 0.5
    M_old, F_M, dF_dM, phi = np.zeros(imax), np.ones(imax), np.zeros(imax), np.zeros(imax)
    
    const1 = 2/(gamma + 1)
    const2 = (gamma - 1)/2
    const3 = (gamma + 1)/(gamma - 1)
    const4 = 2/(gamma - 1)
    const5 = gamma/(gamma - 1)
    
    for i in range(100):  # ?? arbitrary no. of iterations (enough for convergence), not ideal, but had issues with array truth values
        M_old = M_new
        phi = const1 * (1 + const2*(M_old**2))
        F_M = phi**const3 - (A_bar**2)*(M_old**2)
        dF_dM = 2*M_old*(phi**const4 - A_bar**2)
        M_new = M_old - F_M/dF_dM
    
    solution_vector = np.zeros((7,imax))
    solution_vector[0, :] = x #x-position
    solution_vector[1, :] = A #area
    solution_vector[2, :] = M_new #Mach number
    
    psi = 1 + const2*solution_vector[2, :]**2
    
    solution_vector[3,:] = t0/psi #temperature
    solution_vector[4,:] = p0/psi**const5 #pressure
    solution_vector[5,:] = solution_vector[4,:]/(R_air*solution_vector[3,:]) #density
    solution_vector[6,:] = solution_vector[2, :]*(gamma*R_air*solution_vector[3,:])**0.5 #velocity
    
    return solution_vector