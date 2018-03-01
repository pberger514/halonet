import numpy as np
from . import _fast_tools

_DATADIR = "/scratch/p/pen/pberger/train/catalogs/"
_FIELDDIR = "/scratch/p/pen/pberger/train/fields/"

class HaloCatalog(object):

    _data = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Rth', 'xL', 'yL', 'zL', 'zform']

    def to_binary_grid(self, boxsize, ngrid):
        
        """
        dg = float(boxsize)/ngrid
        gloc = np.linspace(-(boxsize/2-dg/2), boxsize/2-dg/2, ngrid)
        gx, gy, gz = np.meshgrid(gloc, gloc, gloc)
        gpos = np.array([gx, gy, gz])

        grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.int8)

        for hi in range(self.Non):
            print hi
            for ii in range(3):
                if ii == 0:
                    halo = (np.absolute(gpos[ii]-self.Lpos[ii, hi]) < self.Rth[hi]).astype(np.int8)
                else:
                    halo += (np.absolute(gpos[ii]-self.Lpos[ii, hi]) < self.Rth[hi]).astype(np.int8)

            grid += (halo == 3).astype(np.int8)
        """

        grid = _fast_tools._to_binary_mask(boxsize, ngrid, self.Lpos, self.Rth)
        
        return grid

    @classmethod
    def from_file(cls, filename, Rcut=None):

        pkfile=open(filename,"rb")

        Non    = np.fromfile(pkfile,dtype=np.int32,count=1)
        RTHmax = np.fromfile(pkfile,dtype=np.float32,count=1)
        zin    = np.fromfile(pkfile,dtype=np.float32,count=1)
        print "\nNumber of halos to read in = ",Non[0], RTHmax, zin

        hc = HaloCatalog()
        
        Non = Non[0]
        hc.Non = Non
        hc.Rthmax = RTHmax
        hc.zin = zin
        
        npkdata = 11*Non
        peakdata = np.fromfile(pkfile, dtype=np.float32, count=npkdata)
        peakdata = np.reshape(peakdata,(Non,11))

        #Apply mass cut if required.
        if Rcut is not None:
            Rth = peakdata[:, 6]
            dm = Rth > Rcut
            peakdata = peakdata[dm]
            hc.Non = int(np.sum(dm))
            print "New Non after cut at Rth of %2.5f is:" % Rcut, hc.Non

        hc.peakdata = {}

        for di, ds in enumerate(cls._data):
            hc.peakdata[ds] = peakdata[:, di]

        return hc

    @property
    def pos(self):
        return np.array([self.peakdata['x'], self.peakdata['y'], self.peakdata['z']])

    @property
    def vel(self):
        return np.array([self.peakdata['vx'], self.peakdata['vy'], self.peakdata['vz']])

    @property
    def Rth(self):
        return self.peakdata['Rth']

    @property
    def Lpos(self):
        return np.array([self.peakdata['xL'], self.peakdata['yL'], self.peakdata['zL']])


def split3d(arr3d, nchunks, nbuff=None, dump_edges=False):
    arr3ds = [arr3d,]
    for di in range(3):
        arr3dsi = []
        for ai in range(len(arr3ds)):
            arr3dsi.extend(split(arr3ds[ai], nchunks,
                                 axis=di, nbuff=nbuff,
                                 dump_edges=dump_edges))
        arr3ds = arr3dsi

    del arr3d
    return arr3ds


def split(arr, nchunks, axis=0, nbuff=None, dump_edges=False):

    if nbuff is None:
        arr_list = np.split(arr, nchunks, axis=axis)
    
    else:
        nsub = (arr.shape[axis] - 2*nbuff)/nchunks
        nmesh = nsub + 2*nbuff
        arr_list = []
        for sli in range(nchunks):
            sl = [slice(None)]*len(arr.shape)
            sl[axis] = slice(sli*nsub, nmesh + sli*nsub)
            arr_list.append(arr[sl])

    if dump_edges:
        arr_list = arr_list[1:-1]
        
    return arr_list
