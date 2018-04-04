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



@cython.boundscheck(False)
cpdef _find_pairs(np.ndarray[dtype=np.float32_t, ndim=2] posi, np.ndarray[dtype=np.float32_t, ndim=1] Rthi, int Noni, np.ndarray[dtype=np.float32_t, ndim=2] posj, np.ndarray[dtype=np.float32_t, ndim=1] Rthj, int Nonj, float pos_cut = 5.0, float R_cut = 0.25):

    cdef int hi, hj

    cdef float dpos, dr

    cdef np.ndarray[np.int32_t, ndim=1] pair_index = np.zeros((Noni), dtype=np.int32)
    cdef np.ndarray[np.int8_t, ndim=1] paired = np.zeros((Nonj), dtype=np.int8)
        
    #Loop through i halos
    for hi in range(Noni):
        
        #Loop through j halos
        for hj in range(Nonj):
            
            if (paired[hj] == 0):
                # Compare positions
                dpos = ((posi[0, hi]-posj[0, hj])**2+(posi[1, hi]-posj[1, hj])**2+(posi[2, hi]-posj[2, hj])**2)**(0.5)
            
                if(dpos <= pos_cut):
                
                    #Compare radii
                    dr = ((Rthi[hi] - Rthj[hj])**2)**(0.5)

                    if ((dr / Rthi[hi]) <= R_cut):
                        # We have a match
                        pair_index[hi] = hj
                        paired[hj] = 1
                    
                        break

    return pair_index


@cython.boundscheck(False)
cpdef _average_inside_radius(float boxsize, int Non, np.ndarray[dtype=np.float32_t, ndim=2] pos, np.ndarray[dtype=np.float32_t, ndim=1] Rth, np.ndarray[dtype=np.float32_t, ndim=3] delta):

    cdef int hi, di, xi, yi, zi, xh, yh, zh
    cdef float gx, gy, gz
    cdef int xl, xr, yl, yr, zl, zr

    cdef int ngrid, cellsize
    cdef int Rthc
    
    cdef int Nin, compval
    cdef float dist

    cdef np.ndarray[np.float32_t, ndim=1] av_arr = np.zeros((Non), dtype=np.float32)

    ngrid = delta.shape[0]
    cellsize = <int>(boxsize / ngrid)

    #Loop through halos
    for hi in prange(Non, nogil=True):
        Rthc = <int>(Rth[hi] / cellsize) + 1
        xh = <int>(pos[0, hi]/cellsize + ngrid/2)
        yh = <int>(pos[1, hi]/cellsize + ngrid/2)
        zh = <int>(pos[2, hi]/cellsize + ngrid/2)
        
        # Find bounding box
        compval = xh - Rthc
        if(compval < 0):
            xl = 0
        else:
            xl = compval
        compval = xh + Rthc
        if (compval > ngrid-1):
            xr = ngrid - 1
        else:
            xr = compval

        compval = yh - Rthc
        if (compval < 0):
            yl = 0
        else:
            yl = compval
        compval = yh + Rthc
        if  (compval > ngrid-1):
            yr = ngrid - 1
        else:
            yr = compval
            
        compval = zh - Rthc
        if (compval < 0):
            zl = 0
        else:
            zl = compval
        compval = zh + Rthc
        if (compval > ngrid-1):
            zr = ngrid - 1
        else:
            zr = compval

        Nin = 0
        # Loop through accumulating average
        for xi in range(xl, xr):
            gx = xi*cellsize + cellsize/2 - boxsize/2
            for yi in range(yl, yr):
                gy = yi*cellsize + cellsize/2 - boxsize/2
                for zi in range(zl, zr):
                    gz = zi*cellsize + cellsize/2 - boxsize/2

                    dist = ((gx-pos[0, hi])**2+(gy-pos[1, hi])**2+(gz-pos[2, hi])**2)**(0.5)
                    if(dist <= Rth[hi]):

                        av_arr[hi] += delta[xi, yi, zi] 
                        Nin = Nin + 1
                    
        if(Nin > 0):
            av_arr[hi] /= Nin

    return av_arr
