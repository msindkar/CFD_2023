# output to tecplot
def soln_write(x, y, imax, jmax, prim):
    import numpy as np
    tec = open('soln.dat', 'w')
    tec.write('TITLE = "2D Euler solver solution"\n')
    tec.write('Variables = "x""y""rho""u""v""P"\n')
    tec.write('ZONE\n')
    tec.write('T = "Title"\n')
    tec.write('I =' + str(imax) + '\n')
    tec.write('J =' + str(jmax) + '\n')
    tec.write('DT = (DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE)\n')
    tec.write('DATAPACKING = BLOCK\n')
    tec.write('VARLOCATION = ([3-6]=CELLCENTERED)\n')
    
    for i in np.arange(imax):
        for j in np.arange(jmax):
            tec.write(str(x[i, j]) + '\n')
            
    for i in np.arange(imax):
        for j in np.arange(jmax):
            tec.write(str(y[i, j]) + '\n')
            
    for i in np.arange(imax - 1):
        for j in np.arange(jmax - 1):
            tec.write(str(prim[i, j, 0]) + '\n')
            
    for i in np.arange(imax - 1):
        for j in np.arange(jmax - 1):
            tec.write(str(prim[i, j, 1]) + '\n')
            
    for i in np.arange(imax - 1):
        for j in np.arange(jmax - 1):
            tec.write(str(prim[i, j, 2]) + '\n')
    
    for i in np.arange(imax - 1):
        for j in np.arange(jmax - 1):
            tec.write(str(prim[i, j, 3]) + '\n')
                    
    tec.close()