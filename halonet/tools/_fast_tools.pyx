"""A few miscellaneous Cython routines to speed up critical operations.
"""

from cython.parallel import prange, parallel
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs 

@cython.boundscheck(False)
cpdef _to_binary_mask(float boxsize, int ngrid, np.ndarray[dtype=np.float32_t, ndim=2] pos, np.ndarray[dtype=np.float32_t, ndim=1] Rth):

    cdef int i, j, k, hi, ii, jj, kk, nsubgrid
    cdef int non
    cdef float gx, gy, gz, dg, dist, Rthi, Vsubgrid, dxc, dyc, dzc, dx, dy, dz

    cdef np.ndarray[np.float32_t, ndim=3] grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)

    dg = boxsize / ngrid

    non = Rth.shape[0]
    
    nsubgrid = 2
    Vsubgrid = 1./nsubgrid**3

    #Loop through halos
    for hi in prange(non, nogil=True):
        Rthi = Rth[hi]
        #Loop through positions
        for i in range(ngrid):
            gx = i*dg + dg/2 - boxsize/2
            if(fabs(gx - pos[0, hi]) < Rthi + dg):
                for j in range(ngrid):
                    gy = j*dg + dg/2 - boxsize/2
                    if(fabs(gy - pos[1, hi]) < Rthi + dg):
                        for k in range(ngrid):
                            gz = k*dg + dg/2 - boxsize/2
                            if(fabs(gz - pos[2, hi]) < Rthi + dg):	
                                dxc = gx - pos[0,hi]
                                dyc = gy - pos[1,hi]
                                dzc = gz - pos[2,hi]

                                dist = (dxc**2 + dyc**2 + dzc**2)**(0.5)

			        #We are possibly inside a halo
                                if(dist < Rthi - dg):
                                    #totally inside halo
                                    grid[i, j, k] = 1	

                                elif grid[i, j, k] != 1. :
                                    #partially inside halo
                                    #split into sub cubes to add to mask
                                    for ii in range(nsubgrid):
                                        for jj in range(nsubgrid):
                                            for kk in range(nsubgrid):
                                                dx = dxc - dg/2 + (ii + 0.5)*dg/nsubgrid			
                                                dy = dyc - dg/2 + (jj + 0.5)*dg/nsubgrid			
                                                dz = dzc - dg/2 + (kk + 0.5)*dg/nsubgrid			
						
                                                dist = (dx**2 + dy**2 + dz**2)**(0.5)  

                                                if((dist < Rthi) & (grid[i, j, k] < 1.)):
                                                    grid[i, j, k] += Vsubgrid

    return np.asarray(grid)
