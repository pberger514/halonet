"""
A few miscellaneous Cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libc.stdio cimport printf

@cython.boundscheck(False)
cpdef _to_binary_mask(float boxsize, int ngrid, np.ndarray[dtype=np.float32_t, ndim=2] pos, np.ndarray[dtype=np.float32_t, ndim=1] Rth):

    cdef int i, j, k, hi, di
    cdef int non
    cdef float gx, gy, gz, dg, dist

    cdef np.ndarray[np.int8_t, ndim=3] grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.int8)

    cdef float pnx, pny, pnz
    cdef int flip
    
    dg = boxsize / ngrid
    
    non = Rth.shape[0]
    
    #Loop through halos
    for hi in prange(non, nogil=True):
    #for hi in range(non):
        
        #Loop through positions
        for i in range(ngrid):
            gx = i*dg + dg/2 - boxsize/2
            if (fabs(gx - pos[0, hi]) < Rth[hi]):
                for j in range(ngrid):
                    gy = j*dg + dg/2 - boxsize/2
                    if (fabs(gy - pos[1, hi]) < Rth[hi]):
                        for k in range(ngrid):
                            
                            gz = k*dg + dg/2 - boxsize/2
                            dist = ((gx - pos[0, hi])**2+(gy - pos[1, hi])**2+(gz - pos[2, hi])**2)**(0.5)

                            if (dist < Rth[hi]):
                                #We are inside a halo
                                grid[i, j, k] = 1


        #Check if we should wrap
        flip = 0
        pnx = pos[0, hi]
        if((pos[0, hi] + Rth[hi]) > boxsize / 2):
            pnx = - boxsize/2 - fabs(boxsize/2 - pos[0, hi])
            flip = 1
        elif((pos[0, hi] - Rth[hi]) < -boxsize / 2):
            pnx = boxsize/2 + fabs(-boxsize/2 - pos[0, hi]) 
            flip = 1
        pny = pos[1, hi]
        if((pos[1, hi] + Rth[hi]) > boxsize / 2):
            pny = - boxsize/2 - fabs(boxsize/2 - pos[1, hi])
            flip = 1
        elif((pos[1, hi] - Rth[hi]) < -boxsize / 2):
            pny = boxsize/2 + fabs(-boxsize/2 - pos[1, hi]) 
            flip = 1
        pnz = pos[2, hi]
        if((pos[2, hi] + Rth[hi]) > boxsize / 2):
            pnz = - boxsize/2 - fabs(boxsize/2 - pos[2, hi])
            flip = 1
        elif((pos[2, hi] - Rth[hi]) < -boxsize / 2):
            pnz = boxsize/2 + fabs(-boxsize/2 - pos[2, hi]) 
            flip = 1

        if (flip == 1):
            #Loop through positions
            for i in range(ngrid):
                gx = i*dg + dg/2 - boxsize/2
                if (fabs(gx - pnx) < Rth[hi]):
                    for j in range(ngrid):
                        gy = j*dg + dg/2 - boxsize/2
                        if (fabs(gy - pny) < Rth[hi]):
                            for k in range(ngrid):
                            
                                gz = k*dg + dg/2 - boxsize/2
                                dist = ((gx-pnx)**2+(gy-pny)**2+(gz-pnz)**2)**(0.5)

                                if (dist < Rth[hi]):
                                    #We are inside a halo
                                    grid[i, j, k] = 1
            
    return np.asarray(grid)
