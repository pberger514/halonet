import numpy as np

class HaloCatalog(object):

    _data = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Rth', 'xL', 'yL', 'zL', 'zform']

    def to_binary_grid(self, boxsize, ngrid, wrap=True):
        
        dg = float(boxsize)/ngrid
        gloc = np.linspace(-(boxsize/2-dg/2), boxsize/2-dg/2, ngrid)
        gx, gy, gz = np.meshgrid(gloc, gloc, gloc)

        grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.int8)

        #Loop through halos
        for hi in self.Non:
            halo = np.logical_and(np.logical_and(gx-self.pos[0, hi] < self.Rth[hi],
                                                 gy-self.pos[1, hi] < self.Rth[hi]),
                                  gz-self.pos[2, hi] < self.Rth[hi]))
            grid = np.where(halo, 1, grid)

        return grid

    @classmethod
    def from_file(cls, filename):

        pkfile=open(filename,"rb")

        Non    = np.fromfile(pkfile,dtype=np.int32,count=1)
        RTHmax = np.fromfile(pkfile,dtype=np.float32,count=1)
        zin    = np.fromfile(pkfile,dtype=np.float32,count=1)
        print "\nNumber of halos to read in = ",Non[0], RTHmax, zin

        hc = HaloCatalog()

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
