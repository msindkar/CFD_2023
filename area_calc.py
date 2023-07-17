def area_normals_calc(imax, jmax, x, y):    
    # AREA CALC
    import time as t
    import numpy as np
    
    st = t.time()
    print("Calculating interface areas and normals...")
    W = 1
    
    A_vert = np.zeros((imax, jmax - 1)) # area of vertical faces
    A_hori = np.zeros((imax - 1, jmax)) # area of horizontal faces
    
    normals_v = np.zeros((imax, jmax - 1, 2)) # normals of vertical   faces nv[:, :, 0] x-normals
    normals_h = np.zeros((imax - 1, jmax, 2)) # normals of horizontal faces 
    
    for j in np.arange(jmax - 1):
        for i in np.arange(imax):
            A_vert[i, j] = W*((x[i, j + 1] - x[i, j])**2 + (y[i, j + 1] - y[i, j])**2)**(1/2)
            
    for j in np.arange(jmax):
        for i in np.arange(imax - 1):
            A_hori[i, j] = W*((x[i + 1, j] - x[i, j])**2 + (y[i + 1, j] - y[i, j])**2)**(1/2)
            
    for j in np.arange(jmax - 1):
        for i in np.arange(imax):
            normals_v[i, j, 0] = (y[i, j + 1] - y[i, j])*W/(A_vert[i, j])
            normals_v[i, j, 1] =-(x[i, j + 1] - x[i, j])*W/(A_vert[i, j])
    
    for j in np.arange(jmax):
        for i in np.arange(imax - 1):
            normals_h[i, j, 0] = (y[i, j] - y[i + 1, j])*W/(A_hori[i, j])
            normals_h[i, j, 1] =-(x[i, j] - x[i + 1, j])*W/(A_hori[i, j])
    
    et = t.time()
    print("Interface areas and normals calculated in: "+ str(et - st) + "s")
    
    return A_vert, A_hori, normals_h, normals_v