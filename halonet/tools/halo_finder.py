import numpy as np
from mask_to_halo import *
import sys

# NEED A NEW VERSION OF PYTHON 2.7 for some of the newer numpy functions used
# I use:  module purge ; module load anaconda2/5.0.1

# Simulation info
# File to test on
seed   = '14027'
if len(sys.argv) > 1: seed = str(sys.argv[1])
bx     = 576.
fz     = 576
nbuff  = 32
data   = np.load('predicted_data/nmodel128_Lbox576.0_'+seed+'.npz')

alatt = bx/fz
fzp   = fz/1   # size of box to use (for testing purposes),
               # because saving all steps and fields can use a lot of memory

# Load in data
delta       = data['delta'][:fzp,:fzp,:fzp]
maskt       = data['mask_true'][:fzp,:fzp,:fzp]
maskp       = data['mask_predicted'][:fzp,:fzp,:fzp]
pphalopos   = (data['halopos']+bx/2)/alatt #position in cells
Rthpp       = data['haloRth']/alatt

xpp = pphalopos[:,0]
ypp = pphalopos[:,1]
zpp = pphalopos[:,2]

#sort peak patch by most massive (not really necessary anymore tho)
iRthmax = Rthpp.argsort()
xpp     = xpp[iRthmax[::-1]]
ypp     = ypp[iRthmax[::-1]]
zpp     = zpp[iRthmax[::-1]]
Rthpp  = Rthpp[iRthmax[::-1]]


# Keep only halos in region of box of interest 
dm    = [(xpp<fzp) & (ypp<fzp) & (zpp<fzp)]

xpp   = xpp[dm]
ypp   = ypp[dm]
zpp   = zpp[dm]
Rthpp = Rthpp[dm]
Mpp   = 4*np.pi/3*Rthpp**3
print "Max Rth PP = ", np.max(Rthpp)

find_halos   = True
#find_halos   = False
if find_halos:

    # FIND HALOS: halo_finder(predicted_mask, probability_cut, list_of_smoothing_scales)
    Psmoothcut = [0.99, 0.9, 0.75, 0.5, 0.4]
    Rsmooth    = [1.,1.,1.,1.,1.]
    Rmax       = [32,16,10,10,10]

    # FIND HALOS
    use_pp_pos = False #do halofinding, but using peak patch locations as the center of regions of interest
    x,y,z,R, maskp_halos = halo_finder(maskp,Psmoothcut,Rsmooth,nbuff,Rmax,use_pp_pos,xpp,ypp,zpp)
    
    # halo positions have been shifted by nbuff, to easily account for buffers in measurements.
    # Shift back here, and transform back to physical units
    # To get a 512^3 periodic box of halos from (0,512Mpc)
    xpp   = (xpp-nbuff)*alatt
    ypp   = (ypp-nbuff)*alatt
    zpp   = (zpp-nbuff)*alatt
    Rthpp = Rthpp*alatt
    
    x     = (x-nbuff)*alatt
    y     = (y-nbuff)*alatt
    z     = (z-nbuff)*alatt
    R     = R*alatt
    mass  = 4*np.pi/3 * R**3

    # Save halos
    if use_pp_pos:     fout = "halo_catalogues/nmodel128_Lbox576.0_use_pp_pos_"+seed+".npz"
    if not use_pp_pos: fout = "halo_catalogues/nmodel128_Lbox576.0_"+seed+".npz"
    
    np.savez(fout,x_pp=xpp, y_pp=ypp, z_pp=zpp, Rth_pp=Rthpp, M_pp=Mpp, x_hn=x, y_hn=y, z_hn=z, Rth_hn=R, M_hn=mass)


if not find_halos:
    #DO MEASURMENTS
    Rmin = 1
    Rmax = 32
    Mbin_edge, Mbin_cent, Mcounts, Rshell, count_shell, Pshell_hist, stdshell_hist = halo_measurements(maskp, Rmin, Rmax, xpp, ypp,zpp,Mpp)

    print "save measurements"
    np.savez("measurements_data/halo_measurements_Pin_std_"+seed+".npz", Mbin_edge=Mbin_edge, Mbin_cent=Mbin_cent, Mcounts=Mcounts, Rshell=Rshell, count_shell=count_shell, Pshell_hist=Pshell_hist, stdshell_hist=stdshell_hist)
