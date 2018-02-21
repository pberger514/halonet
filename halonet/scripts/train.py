from halonet.tools import catalog
from halonet.net import model
from halonet.net import loss

from keras import losses
from keras.optimizers import SGD

import numpy as np
import glob
import datetime

#Redirect stdout to stderr for some reason.
#######################
import sys
sys.stdout = sys.stderr
#######################

# Get the file containing the catalogs and their associated inital conditions
fz = 256 # input field size
sz = 64  # size to input to the network
bx = 568.0 # Box size im Mpc
catalogfiles = np.sort(glob.glob(catalog._DATADIR+'*merge*'))
fieldfiles = np.sort(glob.glob(catalog._FIELDDIR+'*delta*'))
nfiles = len(catalogfiles)

savedir = "/scratch/p/pen/pberger/train/model/"
inputmodelfile = None
inputweightsfile = None
outputmodelfile = savedir + "halonet-%s.h5" % datetime.datetime.now().date()
outputhistoryfile = savedir + "halonethistory-%s.npy" % datetime.datetime.now().date()

filesperbatch_start = 5
nepochs_start = 50

check=False #Output some verification information
reflect_right_away = False
#Batch size info
nlevels=5
nconv = 3
vp=0.125 # Percentage of batch for validation

if inputmodelfile is None:

    if inputweightsfile is None:
        hnet = model.get_model(nlevels, input_shape=(sz, sz, sz, 1),
                               nconv=nconv, dropout=0.5)
        sgd = SGD(lr=.00025, momentum=0.4)
        filesperbatch = filesperbatch_start
        nepochs = nepochs_start
        
    else:
        print "Changing the learning rate, momentum, and nepochs."
        hnet = model.get_model(nlevels, input_shape=(sz, sz, sz, 1), nconv=nconv)
        filesperbatch = nfiles
        nepochs = 10
        sgd = SGD(lr=.1, momentum=0.5)
        #sgd = SGD(lr=.0001, momentum=0.01)
        
    # Compile model
    hnet.compile(loss=loss.dice_loss_coefficient, optimizer=sgd, metrics=['accuracy',])
    #hnet.compile(loss=losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy',])
    
    if inputweightsfile is not None:
        hnet.load_weights(inputweightsfile)
    
else:
    from keras.models import load_model
    hnet = load_model(inputmodelfile,
                      custom_objects={'dice_loss_coefficient':loss.dice_loss_coefficient})

niter = 8 #Number of times to go through full dataset
for it in range(niter):
    for fi in range(nfiles):
        if not ((filesperbatch == nfiles) and (it > 0)):
            # Load in a catalog and associated field
            hc = catalog.HaloCatalog.from_file(catalogfiles[fi])
            deltai = np.fromfile(fieldfiles[fi], dtype=np.float32).reshape(fz, fz, fz).T # fortran -> C ?

            # Renormalize delta to have standard deviation 1.
            std = np.std(deltai)
            deltai = deltai/std
    
            # Compute the binary mask (ground truth)
            maski = hc.to_binary_grid(bx, fz).astype(np.float32)

            if check and fi == 0 and it == 0:
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
        
            if check and fi == 0:
                # Write input data to file to check
                outfile = open('checkdata_delta.dat', 'wb')
                delta[0, :, :, :, 0].tofile(outfile)
                outfile.close()
                outfile = open('checkdata_mask.dat', 'wb')
                mask[0, :, :, :, 0].tofile(outfile)
                outfile.close()
        
        if (fi + 1) % filesperbatch == 0 :

            #if (fi + 1) / filesperbatch == 2 :
            #Change learning rate, momentum, and nepochs
            #    print "Changing the learning rate, momentum, and nepochs."
            #    sgd.lr.set_value(0.0001)
            #    sgd.momentum.set_value(0.5)
            #    nepochs_i = 20
            
            # Okay now train the network
            batch_size = int((1.0-vp)*nchunks**3)*filesperbatch
            test_size = int(vp*nchunks**3)*filesperbatch

            # Separate into training and validation sets
            delta_t = delta[:batch_size]
            mask_t = mask[:batch_size]
        
            delta_v = delta[batch_size:]
            mask_v = mask[batch_size:]

            if nepochs is None:
                nepochs_i = filesperbatch*nchunks**3/8
            else:
                nepochs_i = nepochs

            print "%i, Training on a batch of size %s , %s ..." % (fi, delta_t.shape, mask_t.shape)
            print "With a validation of size %s , %s ..." % (delta_v.shape, mask_v.shape)
            print "With mini-batches of size %i, for %i epochs ..." % (batch_size/filesperbatch/(nchunks**3/8), nepochs_i)

            #Perform random reflections for augmentation
            if it > 0 or reflect_right_away:
                print "Reflecting ..."
                rpc = 0.8 # Percentage to reflect
                rsel = np.sort(np.random.choice(np.arange(batch_size), size=int(batch_size*rpc),
                                                replace=False))
                
                if np.random.uniform() >= 0.6:
                    delta_t[rsel] = delta_t[rsel, ::-1]
                    mask_t[rsel] = mask_t[rsel, ::-1]
                    #delta_v = delta_v[:, ::-1]
                    #mask_v = mask_v[:, ::-1]

                if np.random.uniform() >= 0.6:
                    delta_t[rsel] = delta_t[rsel, :, ::-1]
                    mask_t[rsel] = mask_t[rsel, :, ::-1]

                if np.random.uniform() >= 0.6:
                    delta_t[rsel] = delta_t[rsel, :, :, ::-1]
                    mask_t[rsel] = mask_t[rsel, :, :, ::-1]


            # Okay now actually train the network
            # Reshape mask for voxelwise loss.
            history = hnet.fit(delta_t,
                               np.concatenate([mask_t, 1.0-mask_t], axis=-1),
                               validation_data=(delta_v,
                                                np.concatenate([mask_v, 1.0-mask_v], axis=-1)),
                               epochs=nepochs_i,
                               batch_size=batch_size/filesperbatch/(nchunks**3/8),
                               verbose=2)

            #Save the model
            hnet.save(outputmodelfile)
            np.save(outputhistoryfile, history.history)
        
            #Reshape mask for voxelwise loss.
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
