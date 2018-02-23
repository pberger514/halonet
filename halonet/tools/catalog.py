import numpy as np

_DATADIR = "/scratch2/p/pen/pberger/ppruns/train/trainv0/output/"
_FIELDDIR = "/scratch2/p/pen/pberger/ppruns/train/trainv0/fields/"

class HaloCatalog(object):

    _data = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Rth', 'xL', 'yL', 'zL', 'zform']

    def to_binary_grid(self, boxsize, ngrid):

        import _fast_tools
        
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
    def from_file(cls, filename):

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

def split3d(arr3d, nchunks):
    arr3ds = [arr3d,]
    for di in range(3):
        arr3dsi = []
        for ai in range(len(arr3ds)):
            arr3dsi.extend(np.split(arr3ds[ai], nchunks, axis=di))
        arr3ds = arr3dsi

    del arr3d
    return arr3ds
