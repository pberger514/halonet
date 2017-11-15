"""A few miscellaneous Cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs 

@cython.boundscheck(False)
cpdef _to_binary_mask(float boxsize, int ngrid, np.ndarray[dtype=np.float32_t, ndim=2] pos, np.ndarray[dtype=np.float32_t, ndim=1] Rth):

    cdef int i, j, k, hi
    cdef int non
    cdef float gx, gy, gz, dg, dist

    cdef np.ndarray[np.int8_t, ndim=3] grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.int8)

    dg = boxsize / ngrid

    non = Rth.shape[0]
    
    #Loop through halos
    for hi in prange(non, nogil=True):
        #Loop through positions
        for i in range(ngrid):
            gx = i*dg + dg/2 - boxsize/2
            if(fabs(gx - pos[0, hi]) < Rth[hi]):
                for j in range(ngrid):
                    gy = j*dg + dg/2 - boxsize/2
                    if(fabs(gy - pos[1, hi]) < Rth[hi]):
                        for k in range(ngrid):

                            gz = k*dg + dg/2 - boxsize/2
                            dist = ((gx - pos[0, hi])**2+(gy - pos[1, hi])**2+(gz - pos[2, hi])**2)**(0.5)

                            if(dist < Rth[hi]):
                                #We are inside a halo
                                grid[i, j, k] = 1

    return np.asarray(grid)
