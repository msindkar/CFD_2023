def mesh_loader():

    # ---------- ---------- ---------- ----------
    # Loads mesh, currently 3D only
    # Returns in order: imax, jmax, arrays of node x coordinates and y coordinates
    # Must be in same directory as config file
    # ---------- ---------- ---------- ----------    
    
    from config import gridname
    import numpy as np
    import time as t
    from shapely import Polygon, centroid
    
    st = t.time()
    
    print('Loading mesh...')
    
    f = open(gridname, 'r')
    lines = f.readlines()
    f.close()
    
    str_size = lines[1].split()
    grid_size = int(str_size[0])*int(str_size[1])*int(str_size[2])
    
    imax = int(str_size[0])
    jmax = int(str_size[1])
    
    coord_block = int(grid_size/4) + 1
    
    x = 0
    y = 0
    
    #--- for Y
    
    for i in range(len(lines) - 1 - coord_block, len(lines) - 1 - 2*coord_block, -1):
        splitline = lines[i].split()
        splitline.reverse()
        splitline = [float(j) for j in splitline]
        y = np.append(y, splitline)
    
    y = np.delete(y, np.arange(0, int((grid_size/2) + 1))) # REMEMBER APPENDED TO ZERO
    y = y.reshape((imax, jmax))
    y = np.transpose(y)
    
    #--- for X
    
    for i in range(len(lines) - 1 - 2*coord_block, 1, -1):
        splitline = lines[i].split()
        splitline.reverse()
        splitline = [float(j) for j in splitline]
        x = np.append(x, splitline)
        
    x = np.delete(x, np.arange(0, int((grid_size/2) + 1))) # REMEMBER APPENDED TO ZERO
    x = x.reshape((imax, jmax))
    x = np.transpose(x)
    
    x_cell = np.zeros((imax - 1, jmax - 1))
    y_cell = np.zeros((imax - 1, jmax - 1))

    for j in np.arange(jmax - 1):
        for i in np.arange(imax - 1):
            x_cell[i, j] = centroid(Polygon([(x[i, j], y[i, j]), (x[i + 1, j], y[i + 1, j]), (x[i + 1, j + 1], y[i + 1, j + 1]), (x[i, j + 1], y[i, j + 1])])).x
            y_cell[i, j] = centroid(Polygon([(x[i, j], y[i, j]), (x[i + 1, j], y[i + 1, j]), (x[i + 1, j + 1], y[i + 1, j + 1]), (x[i, j + 1], y[i, j + 1])])).y
    
    et = t.time()
    
    print('Mesh loaded in: ' + str((et - st)) + 's')
    
    return imax, jmax, x, y, x_cell, y_cell