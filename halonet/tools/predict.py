import numpy as np
from halonet.net import model
from halonet.net import loss
from halonet.tools import catalog
import sys
import time

# File to test on
nmodel = 128
bx     = 576.
fz     = 576
nbuff  = 32
ntile  = 8
Rcut   = 1.86105

# Load model & weights. If this works we're good
weightsfile = '/scratch2/p/pen/pberger/ppruns/train/trained_models/halonetv2-2018-03-08-nobnthird-nodo.h5'
hnet        = model.get_model(5,nconv=3,input_shape=(nmodel,nmodel,nmodel,1),lrelu_alpha=0.05,batch_norm=False) 

# Default hyperparameters since we dont care about training
hnet.compile(loss=loss.dice_loss_coefficient, optimizer='sgd')
hnet.load_weights(weightsfile)


#Now predict for a given simulation
seed   = '14027'
if len(sys.argv) > 1: seed = str(sys.argv[1])

print "\npredicting on seed", seed
catalogfile = '/scratch2/p/pen/pberger/ppruns/train/trainv2/output/testdata/512Mpc_n512_nb32_nt2_merge.pksc.'+seed
densityfile = '/scratch2/p/pen/pberger/ppruns/train/trainv2/fields/testdata/'+seed+'.deltawrap'

# Load in density field to predict on
delta       = np.fromfile(densityfile,dtype=np.float32).reshape(fz,fz,fz,order="F")
sigma_delta = np.std(delta[nbuff:-nbuff, nbuff:-nbuff, nbuff:-nbuff])

# Load in a catalog and associated field                        
hc          = catalog.HaloCatalog.from_file(catalogfile, Rcut=Rcut)

# Compute the binary mask (ground truth)                                                                                 
maskt       = hc.to_binary_grid(bx, fz).astype(np.float32)
#maskt_large_Rth      = hc.to_binary_grid(bx, fz, 'Rth').astype(np.float32)
#maskt_large_Rthfrac  = hc.to_binary_grid(bx, fz, 'Rthfrac').astype(np.float32)

# Cut density field down to desired size (list of 64**3)
delta_all   = catalog.split3d(delta, ntile, nbuff=nbuff)

# Predict mask for this density field
maskp = np.zeros((fz,fz,fz))
nsub  = (fz-2*nbuff)/ntile
nmesh = nsub+2*nbuff


# as we need to treat edges different than the central regions (want a fx**3 predicted mask, not (fz-nbuff*2)**3
tlist_keep     = np.arange(ntile+1)*nsub + nbuff
tlist_keep[0]  = 0
tlist_keep[-1] = fz

boxnum = 0

start = time.time()
for i in range(ntile):
    i_lbuff = nbuff
    i_rbuff = -nbuff
    if (i == 0):       i_lbuff = 0     #keep outer buffer
    if (i == ntile-1): i_rbuff = nmesh #keep outer buffer

    for j in range(ntile):
        j_lbuff = nbuff
        j_rbuff = -nbuff
        if (j == 0):       j_lbuff = 0     #keep outer buffer
        if (j == ntile-1): j_rbuff = nmesh #keep outer buffer
        for k in range(ntile):
            k_lbuff = nbuff
            k_rbuff = -nbuff
            if (k == 0):       k_lbuff = 0     #keep outer buffer
            if (k == ntile-1): k_rbuff = nmesh #keep outer buffer

            if (boxnum % ntile**2) == 0: print 'predicting box number = ', boxnum
            maskpi         = hnet.predict(delta_all[i*ntile**2+j*ntile+k].reshape(1, nmodel, nmodel, nmodel, 1)/sigma_delta)
            maskp[ tlist_keep[i]:tlist_keep[i+1], tlist_keep[j]:tlist_keep[j+1], tlist_keep[k]:tlist_keep[k+1]] = np.array(maskpi[0, i_lbuff:i_rbuff, j_lbuff:j_rbuff, k_lbuff:k_rbuff, 0])

            boxnum +=1

end = time.time()
print 'Predicting the mask took ',end-start,' seconds'

print "shape of the output masks [density, maskt, maskp] = ", delta.shape, maskt.shape, maskp.shape

#np.savez(str(nmodel)+'_'+str(seed)+'_fields_all',delta=delta_large, mask_true=maskt_large, mask_true_Rthfrac=maskt_large_Rthfrac, mask_true_Rth=maskt_large_Rth, mask_predicted=maskp, halopos=hc.Lpos.T, haloRth=hc.Rth)
np.savez('predicted_data/nmodel'+str(nmodel)+'_Lbox'+str(bx)+'_'+str(seed),delta=delta, mask_true=maskt,  mask_predicted=maskp, halopos=hc.Lpos.T, haloRth=hc.Rth)
