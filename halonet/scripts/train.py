from .tools import catalog
from .net import model
from .net import loss

import glob

nlevels=4
vp=0.125 # Percentage of batch for validation
hnet = model.get_model(nlevels)

# Compile model
hnet.compile(loss=loss.dice_loss_coefficient, optimizer='sgd', metrics=['accuracy',])

# Get the file containing the catalogs and their associated inital conditions
fz = 256 # input field size
sz = 64  # size to input to the network
bx = 512.0 # Box size im Mpc
catalogfiles = np.sort(glob.glob(catalog._DATADIR+'*merge*'))
fieldfiles = np.sort(glob.glob(catalog._FIELDDIR+'Fvec*'))

nfiles = len(catalogfiles)

for fi in range(nfiles):
    # Create a batch from a single file
    # Load in a catalog and associated field
    hc = catalog.HaloCatalog.from_file(catalogfiles[fi])
    delta = np.fromfile(fieldfiles[fi], dtype=np.float32).reshape(fz, fz, fz)
    
    # Compute the binary mask (ground truth)
    mask = hc.to_binary_grid(bx, fz)

    # Split large arrays into sub-chunks
    nchunks = fz/sz
    delta = np.array(split3d(delta, nchunks))
    mask = np.array(split3d(mask, nchunks))
    
    # We now have 4D arrays of shape (batch_size, sz, sz, sz)
    batch_size = int((1.0-vp)*nchunks**3)
    test_size = int(vp*nchunks**3)

    # Okay now train the network
    train_loss, train_acc = hnet.train_on_batch(delta[:batch_size], mask[:batch_size])
    test_loss, test_acc = hnet.test_on_batch(delta[batch_size:], mask[batch_size:])

    if (fi == 0) or (fi % 10 == 0): 
        print fi, train_loss, train_acc, test_loss, test_acc


def split3d(arr3d, nchunks):
    arr3ds = [arr3d,]
    for di in range(3):
        arr3dsi = []
        for ai in range(len(arr3ds)):
            arr3dsi.extend(np.split(arr3ds[ai], nchunks, axis=di))
        arr3ds = arr3dsi

    del arr3d
    return arr3ds
