def test_mesh():
    # TEST WITH CARTESIAN MESH
    import numpy as np
    
    imax = 9
    jmax = 9
    
    dx = 1.0
    dy = 1.0
    
    x = np.zeros((imax, jmax))
    y = np.zeros((imax, jmax))
    
    a = np.zeros((1, jmax))
    b = np.zeros((imax))
    
    for i in np.arange(imax):
        x[i, :] = a[:, :]
        a[:, :]+= np.ones((1, jmax))*dx
        
    for j in np.arange(jmax):
        y[:, j] = b[:]
        b[:]+= np.ones((imax))*dy
        
    x_cell = np.zeros((imax - 1, jmax - 1))
    y_cell = np.zeros((imax - 1, jmax - 1))
    
    a_cell = np.ones((1, jmax - 1))*0.5
    b_cell = np.ones((imax - 1))*0.5
        
    for k in np.arange(imax - 1):
        x_cell[k, :] = a_cell[:, :]
        a_cell[:, :]+= np.ones((1, jmax - 1))*dx
        
    for l in np.arange(jmax - 1):
        y_cell[:, l] = b_cell[:]
        b_cell[:]+= np.ones((imax - 1))*dy
    
    return imax, jmax, x, y, x_cell, y_cell