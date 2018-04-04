import numpy as np
import numpy.linalg as la
from . import _fast_tools

_DATADIR = "/scratch/p/pen/pberger/train/catalogs/"
_FIELDDIR = "/scratch/p/pen/pberger/train/fields/"

class HaloCatalog(object):

    _data = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Rth', 'xL', 'yL', 'zL', 'zform']

    def to_binary_grid(self, boxsize, ngrid):
        
        """
        Return a ngrid**3 grid with 1s at grid points with centres
        inside Lagrangian positions of halos and 0s elsewhere.

        Parameters
        ----------
        boxsize : float
            Side length of box in Mpc

        ngrid : int
            Number of position points to bin to.
        
        Returns
        -------
        grid : np.ndarray([ngrid, ngrid, ngrid])
            The binary grid.

        """

        grid = _fast_tools._to_binary_mask(boxsize, ngrid, self.Lpos, self.Rth)
        
        return grid

    def do_measurements(self, boxsize, delta, verbose=False):
        """
        Here we expect a periodic delta as input.

        """

        ngrid = delta.shape[0]

        cellsize = boxsize / ngrid

        Rth_max = int(np.ceil(self.Rthmax/cellsize))

        # Array of k values
        k = np.fft.fftfreq(ngrid, d=cellsize)
        klist = [k[:, np.newaxis, np.newaxis], k[np.newaxis, :, np.newaxis], k[np.newaxis, np.newaxis, :]]
        k2 = klist[0]**2 + klist[1]**2 + klist[2]**2
        
        # Create container
        strain_bar = np.zeros((self.Non, 3, 3), dtype=np.float32)
        
        #Loop through elements of strain tensor
        for ii in range(3):
            for jj in range(3):

                if ii <= jj:

                    if verbose:
                        print "Performing 3D fft for i=%i, j=%i..." % (ii, jj)
                    dij = np.fft.ifftn(np.fft.fftn(delta)*(-klist[ii]*klist[jj]/k2)).real.astype(np.float32)

                    if verbose:
                        print "Averaging in spheres..."
                    strain_bar[:, ii, jj] = _fast_tools._average_inside_radius(boxsize, self.Non, self.Lpos, self.Rth, dij)

        if verbose:
            print "Performing eigendecomposition..."
        # Get eigenvectors and eigenvalues
        w, v = la.eigh(strain_bar, UPLO='U')

        if verbose:
            print "Getting average delta..."
        #Also get the average delta
        delta_bar = _fast_tools._average_inside_radius(boxsize, self.Non, self.Lpos, self.Rth, delta)

        # Normalize eigenvalues
        w /= (delta_bar[:, np.newaxis]/3)

        # Get the good stuff
        meas = np.zeros((self.Non, 3))

        if verbose:
            print "Computing measurements..."
        meas[:, 0] = delta_bar
        meas[:, 1] = (w[:,-1] - w[:,-3])/6
        meas[:, 2] = (w[:,-1] - 2*w[:,-2] + w[:,-3])/6

        return meas

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
