from halonet.tools import catalog
from halonet.net import model
from halonet.net import loss

from keras import losses
from keras.optimizers import SGD

import numpy as np
import glob

# Get the file containing the catalogs and their associated inital conditions
fz = 256 # input field size
sz = 64  # size to input to the network
bx = 512.0 # Box size im Mpc
filesperbatch = 5
catalogfiles = np.sort(glob.glob(catalog._DATADIR+'*merge*'))
fieldfiles = np.sort(glob.glob(catalog._FIELDDIR+'*delta*'))
#Batch size info

nlevels=4
vp=0.125 # Percentage of batch for validation
hnet = model.get_model(nlevels, input_shape=(sz, sz, sz, 1),
                       lrelu_alpha=0.05, dropout=True)

sgd = SGD(lr=.1, momentum=0.99)
niter = 10

# Compile model
#hnet.compile(loss=loss.dice_loss_coefficient, optimizer='sgd', metrics=['accuracy',])
#hnet.compile(loss=loss.dice_loss_coefficient, optimizer=sgd, metrics=['accuracy',])
#hnet.compile(loss=loss.dice_loss_coefficient, optimizer=sgd)
hnet.compile(loss=losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy',])
#hnet.compile(loss=losses.binary_crossentropy, optimizer='sgd', metrics=['accuracy',])


print catalogfiles[:10], fieldfiles[:10]

nfiles = len(catalogfiles)

for fi in range(nfiles):
    # Load in a catalog and associated field
    hc = catalog.HaloCatalog.from_file(catalogfiles[fi])
    deltai = np.fromfile(fieldfiles[fi], dtype=np.float32).reshape(fz, fz, fz).T # fortran -> C ?

    # Renormalize delta to have standard deviation 1.
    std = np.std(deltai)
    print 'delta has std:', std
    deltai = deltai/std**2
    
    # Compute the binary mask (ground truth)
    maski = hc.to_binary_grid(bx, fz).astype(np.float32)

    if fi == 0:
        # Write input data to file to check
        outfile = open('checkdata_delta_large.dat', 'wb')
        deltai.tofile(outfile)
        outfile.close()

        outfile = open('checkdata_mask_large.dat', 'wb')
        maski.tofile(outfile)
        outfile.close()

    # Split large arrays into sub-chunks
    nchunks = fz/sz
    deltai = np.array(catalog.split3d(deltai, nchunks))
    deltai = deltai.reshape(nchunks**3, sz, sz, sz, 1)
    maski = np.array(catalog.split3d(maski, nchunks))
    maski = maski.reshape(nchunks**3, sz, sz, sz, 1)
    
    # We now have 5D arrays of shape (nchunks**3, sz, sz, sz, nchannels=1)
    # Concatenate them on to the batch
    if (fi % filesperbatch) == 0:
        delta = deltai
        mask = maski
    else:
        delta = np.concatenate((deltai, delta), axis=0)
        mask = np.concatenate((maski, mask), axis=0)
        
    if fi == 0:
        # Write input data to file to check
        outfile = open('checkdata_delta.dat', 'wb')
        delta[0, :, :, :, 0].tofile(outfile)
        outfile.close()

        outfile = open('checkdata_mask.dat', 'wb')
        mask[0, :, :, :, 0].tofile(outfile)
        outfile.close()
        
    if (fi + 1) % filesperbatch == 0 :
        
        # Okay now train the network
        batch_size = int((1.0-vp)*nchunks**3)*filesperbatch
        test_size = int(vp*nchunks**3)*filesperbatch

        # Separate into training and validation sets
        delta_t = delta[:batch_size]
        mask_t = mask[:batch_size]
        
        delta_v = delta[batch_size:]
        mask_v = mask[batch_size:]

        print fi, "Training on a batch of size %s , %s ..." % (delta_t.shape, mask_t.shape)
        print "With a validation of size %s , %s ..." % (delta_v.shape, mask_v.shape)        

        """
        #Perform reflections for augmentation
        for di in range(6):
            if di % 3 == 0:
                delta_t = delta_t[:, ::-1]
                mask_t = mask_t[:, ::-1]
                delta_v = delta_v[:, ::-1]
                mask_v = mask_v[:, ::-1]
            if di % 3 == 1:
                delta_t = delta_t[:, :, ::-1]
                mask_t = mask_t[:, :, ::-1]
                delta_v = delta_v[:, :, ::-1]
                mask_v = mask_v[:, :, ::-1]
            if di % 3 == 2:
                delta_t = delta_t[:, :, :, ::-1]
                mask_t = mask_t[:, :, :, ::-1]
                delta_v = delta_v[:, :, :, ::-1]
                mask_v = mask_v[:, :, :, ::-1]
        """
    
        # Okay now actually train the network
        # Reshape mask for voxelwise loss.
        for iter in range(niter):
            hnet.fit(delta_t, np.concatenate([mask_t, 1.0-mask_t], axis=-1),
                     validation_data=(delta_v, np.concatenate([mask_v, 1.0-mask_v], axis=-1)),
                     epochs=filesperbatch*nchunks**3/8,
                     batch_size=batch_size/filesperbatch/(nchunks**3/8),
                     verbose=2)

            #Reshape mask for voxelwise loss.
            #for it in range(niter):
            #    train_loss, train_acc = hnet.train_on_batch(delta_t,
            #                                                mask_t.reshape(batch_size, -1))

        #test_loss, test_acc = hnet.test_on_batch(delta_v,
        #                                         mask_v.reshape(test_size, -1))

        #print fi, train_loss, train_acc, test_loss, test_acc


        #Shuffle arrays along batch axis
        #rng_state = np.random.get_state()
        #np.random.shuffle(delta_ts)
        #np.random.set_state(rng_state)
        #np.random.shuffle(mask_ts)

        
                

            
        



    
        
