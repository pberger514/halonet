import numpy as np
from scipy import ndimage

def find_connected_regions(mask,Pcut,Mrange):
    '''
    Function to find connected regions above a threshold of Pcut in a 3D mask
    Returns positon (x,y,z), size, and connected regions found
    '''
    # Links used:
    # https://stackoverflow.com/questions/38598690/how-to-find-the-diameter-of-objects-using-image-processing-in-python
    # https://stackoverflow.com/questions/33707095/how-to-locate-a-particular-region-of-values-in-a-2d-numpy-array?answertab=active#tab-top

    blobsi = mask > Pcut

    # label connected regions that satisfy pin > Pcut
    labelsi, nlabelsi = ndimage.label(blobsi)
        
    # Find the mass of each connected region in grid units
    halo_indi, massi = np.unique(labelsi[labelsi>0], return_counts=True)

    # Find the centers of mass of the connected regions
    xi, yi, zi =  np.vstack(ndimage.center_of_mass(mask, labelsi, np.arange(nlabelsi) + 1)).T

    # Cut proto-halos not in mass range
    # Probably not needed, but here just incase extremely large regions get connected
    dm        = [(massi >= 1) & (massi < Mrange)]
    xi        = xi[dm]
    yi        = yi[dm]
    zi        = zi[dm]
    massi     = massi[dm]

    return xi,yi,zi,massi


def smooth_field(fieldin,rsmooth):
    '''
    function to smooth an nD field using a gaussian smoothing kernel on scale rsmooth
    '''
    return ndimage.gaussian_filter(fieldin,rsmooth)



def halo_finder(mask,Psmoothcut,Rsmooth,nbuff,Rmax,use_pp_pos,xpp,ypp,zpp):
    '''
    Function to perform halo finding in a 3D probability field
    1.) Smooths input mask 
    2.) Finds connected regions in the smoothed field above probability threshold Psmoothcut
    3.) Using the center of each connected probability region above the cutoff, finds halos by going out from the
        center in spherical shells, until the mean probability of the shell has dropped below a probability threshold Pshellcut.
        The radius of this shell is then assigned as the radius of the halo, and the position and size of the halo is 
        appended to the master list
    4.) Remove probability cells from the mask included inside the halos just found, and repeat steps 1-4 on increasingly small filter scales 
    '''
    n        = mask.shape[0]
    mask_out = mask*1.
    
    nsmooth = len(Rsmooth)
    if use_pp_pos: nsmooth=1
    # arrays to save mask at various stages for debugging
    #mask_out_all   = np.zeros((mask_out.shape[0],mask_out.shape[1],mask_out.shape[2],nsmooth))
    #mask_sm_all    = np.zeros((mask_out.shape[0],mask_out.shape[1],mask_out.shape[2],nsmooth))
    mask_found = np.zeros((mask_out.shape[0],mask_out.shape[1],mask_out.shape[2]))
    #set probability threshold
    Mcut = 1e6

    Rmin = np.sqrt(3) #Radius of minimum sized halo (27 cell halo), in units of cells

    x        = np.array([])
    y        = np.array([])
    z        = np.array([])
    Rhalo    = np.array([])

    for i in range(nsmooth):
#    for i in range(1): # for testing purposes

        print "running for Psmoothcut[i] = ", Psmoothcut[i]
        Rmaxi = Rmax[i] 

        mask_sm = smooth_field(mask_out, Rsmooth[i])    

        # Skip filter if there is no probability > Psmoothcut
        if len(mask_sm[mask_sm > Psmoothcut[i]]) == 0: continue

        if not use_pp_pos:
            # To get the central location of a desired halo, by finding the center of a blob
            # above a probability threshold, after smoothing the mask on a scale Rsmooth[i]
            x_sm, y_sm, z_sm, mass_sm  = find_connected_regions(mask_sm,Psmoothcut[i],Mcut)
            # keep if further than nbuff cells from boxedge
            dm = [ (x_sm > nbuff)  & (y_sm > nbuff)  & (z_sm > nbuff) &
                   (x_sm < n-nbuff)  & (y_sm < n-nbuff)  & (z_sm < n-nbuff) ]
            x_sm        = x_sm[dm]
            y_sm        = y_sm[dm]
            z_sm        = z_sm[dm]
            mass_sm     = mass_sm[dm]

        if use_pp_pos:
            x_sm = xpp
            y_sm = ypp
            z_sm = zpp
#        print "potential halos are at: ", np.c_[x_sm,y_sm,z_sm]

        # Now want to calculate the actual size of the halo at these positions by a spherical sum
        # once the average probability in a sphere drops below a certain threshold, that is what we consider a halo

        # set up spherical shell indexes and distances (in smoothing loop for now incase we want smoothing scale-dependent Rmax)
        n_hi       = Rmaxi*2 + 1
        XX, YY, ZZ = np.meshgrid(np.linspace(0,n_hi-1,n_hi),np.linspace(0,n_hi-1,n_hi),np.linspace(0,n_hi-1,n_hi)) 
        Rcell      = np.sqrt( (XX-Rmaxi)**2+ + (YY-Rmaxi)**2 + (ZZ-Rmaxi)**2)

        Rcell_1d   = Rcell.flatten()

        dm_Rcell   = [Rcell_1d <= Rmaxi]
        Rcell_1d   = Rcell_1d[dm_Rcell]
        
        Rcell_ind  = Rcell_1d.argsort()            
        Rcell_1d   = Rcell_1d[Rcell_ind]

        # find unique distances (shells)
        Rshell, count_shell = np.unique(Rcell_1d, return_counts=True)

        # piecewise polynomial best fit
        m1 = 0.04   ; b1 = 0.17
        m2 = 0.0033 ; b2 = 0.4
        xcrossover = 6.5

        Pshellcut  = m1*Rshell + b1
        Pshellcut[Rshell > xcrossover] = m2*Rshell[Rshell > xcrossover]+b2

        # find halo by getting probability shells around each potential position 
        for hi in range(len(x_sm)):

            if hi % 100 == 0: print "Done halo # ", hi,' of Ntotal=',len(x_sm) 

            xii = int(x_sm[hi]) # index of central cell 
            yii = int(y_sm[hi]) # index of central cell 
            zii = int(z_sm[hi]) # index of central cell 

            # get subselection of cells within R <= Rmax
            xmin = xii - Rmaxi ; xmax = xii + Rmaxi + 1
            ymin = yii - Rmaxi ; ymax = yii + Rmaxi + 1
            zmin = zii - Rmaxi ; zmax = zii + Rmaxi + 1

            mask_hi = mask_out[xmin:xmax,ymin:ymax,zmin:zmax] # now have square cutout around peak
            
            
            mask_hi_1d = mask_hi.flatten()
            mask_hi_1d = mask_hi_1d[dm_Rcell]

            # sort by distance from central voxel
            mask_hi_1d = mask_hi_1d[Rcell_ind]


            # dumb way for now to get sum of probability in shell (remove loop later for speed)
            mask_hi_shell = np.zeros(Rshell.shape) # make empty array
            for s in range(len(mask_hi_shell)):
                mask_hi_shell[s] = np.sum(mask_hi_1d[Rcell_1d == Rshell[s]])

            pshell = mask_hi_shell/count_shell

            #find index of first time pshell drops below Pshellcut
            ind_shell = np.argmax(pshell < Pshellcut)
            Rhaloi    = 0.
            if (ind_shell == 0) & (pshell[0] > 0.99): 
                print "halo larger than bounding box, setting Rhalo = Rmax_halo = ",Rmaxi
                Rhaloi = Rmaxi
            elif (ind_shell > 0):
                # set Rhalo to shell smaller than first drop below threshold
                Rhaloi = Rshell[ind_shell-1] 

            if Rhaloi >= Rmin:
#                print "Rhaloi = ", Rhaloi
                # Set cells used in this halo to 0 probabiliy
                mask_out[xmin:xmax,ymin:ymax,zmin:zmax][Rcell < Rhaloi] = 0.
                mask_found[xmin:xmax,ymin:ymax,zmin:zmax][Rcell < Rhaloi] = i+1
                # save halo to master list
                x     = np.append(x,xii+0.5)
                y     = np.append(y,yii+0.5)
                z     = np.append(z,zii+0.5)
                Rhalo = np.append(Rhalo,Rhaloi)
                
            # plot pshell
            #plt.plot(dist_u,pshell)
            #plt.xlabel("radius [cells]")
            #plt.ylabel(r"$P_{in}^{shell}$")
            #plt.savefig("Images/pshell_vs_r.pdf")
            #plt.show()
                
#        mask_out_all[:,:,:,i] = mask_out 
#        mask_sm_all[:,:,:,i]  = mask_sm 
        
    # sort output catalogue by mass
    Rhalo_ind  = Rhalo.argsort()[::-1]
    Rhalo      = Rhalo[Rhalo_ind]
    x          = x[Rhalo_ind]
    y          = y[Rhalo_ind]
    z          = z[Rhalo_ind]
    #    print " x,y,z,Rhalo = \n ", np.c_[x,y,z,Rhalo]

    return x,y,z,Rhalo,mask_found



def halo_measurements(mask,Rmin,Rmax,xin,yin,zin,Massin):
    '''
    Function to perform halo measurements in a 3D probability field
    1.) Using the input coordinates as the center, go out in spherical shells and do measurements
    '''
    n        = mask.shape[0]

    # keep if further than nbuff cells from boxedge
    dm = [ (xin > nbuff)  & (yin > nbuff)  & (zin > nbuff) &
           (xin < n-nbuff)  & (yin < n-nbuff)  & (zin < n-nbuff) ]

    xin        = xin[dm]
    yin        = yin[dm]
    zin        = zin[dm]
    Massin     = Massin[dm]

    Num_halos = len(xin)
    # calculate measurments across spherical shells
    
    # set up spherical shell indexes and distances (in smoothing loop for now incase we want smoothing scale-dependent Rmax)
    n_hi       = Rmax*2 + 1
    XX, YY, ZZ = np.meshgrid(np.linspace(0,n_hi-1,n_hi),np.linspace(0,n_hi-1,n_hi),np.linspace(0,n_hi-1,n_hi)) 
    Rcell      = np.sqrt( (XX-Rmax)**2+ + (YY-Rmax)**2 + (ZZ-Rmax)**2)
    
    Rcell_1d   = Rcell.flatten()
    
    dm_Rcell   = [Rcell_1d <= Rmax]
    Rcell_1d   = Rcell_1d[dm_Rcell]
    
    Rcell_ind  = Rcell_1d.argsort()            
    Rcell_1d   = Rcell_1d[Rcell_ind]
    
    # find unique distances (shells)
    Rshell, count_shell = np.unique(Rcell_1d, return_counts=True)

    mask_hi_shell = np.zeros(Rshell.shape) # make empty array
    std_shell     = np.zeros(Rshell.shape) # make empty array

    Num_shells    = len(Rshell)

    Mmin          = 27
    Mmax          = 4*np.pi/3 * Rmax**3

    nbinsR        = Num_shells
    nbinsM        = 50

    Mbin_edge     = np.logspace(np.log10(27),np.log10(Mmax),nbinsM+1)
    Mbin_cent     = 10**((np.log10(Mbin_edge[1:]) + np.log10(Mbin_edge[:-1]))/2)

    M0            = np.log10(Mbin_edge[0])
    dlogM         = np.log10(Mbin_edge[1]) - np.log10(Mbin_edge[0])

    Mcounts       = np.zeros(nbinsM)
    Pshell_hist   = np.zeros((nbinsM,nbinsR))
    stdshell_hist = np.zeros((nbinsM,nbinsR))

    np.savez("count_shell", count_shell=count_shell)
    # set up grid for adding measurement probabilities to
    # find halo by getting probability shells around each potential position 
    for hi in range(Num_halos):
        xii = int(xin[hi]) # index of central cell 
        yii = int(yin[hi]) # index of central cell 
        zii = int(zin[hi]) # index of central cell 

        Mass_hi = Massin[hi]

        # get subselection of cells within R <= Rmax
        xmin = xii - Rmax ; xmax = xii + Rmax + 1
        ymin = yii - Rmax ; ymax = yii + Rmax + 1
        zmin = zii - Rmax ; zmax = zii + Rmax + 1
        
        mask_hi = mask[xmin:xmax,ymin:ymax,zmin:zmax] # now have square cutout around point
            
        mask_hi_1d = mask_hi.flatten()
        mask_hi_1d = mask_hi_1d[dm_Rcell]
        
        # sort by distance from central voxel
        mask_hi_1d = mask_hi_1d[Rcell_ind]
                
        # dumb way for now to get sum of probability in shell (remove loop later for speed)
        mask_hi_shell *= 0.
        std_shell     *= 0.
        for s in range(Num_shells):
            cells_s          = mask_hi_1d[Rcell_1d == Rshell[s]]
            mask_hi_shell[s] = np.sum(cells_s)
            std_shell[s]     = np.std(cells_s)

        pshell = mask_hi_shell/count_shell

#        Rshell_hist = np.append(Rshell_hist,Rshell)
#        Pshell_hist = np.append(Pshell_hist,pshell)
#        Mshell_hist = np.append(Mshell_hist,pshell*0.+Mass_hi)

        Mind           = int( (np.log10(Mass_hi)-M0) /dlogM) 
        Mcounts[Mind] += 1 

        Pshell_hist[Mind,:]   += pshell
        stdshell_hist[Mind,:] += std_shell

        if hi % 1000 == 0: print "Done halo # ", hi," in range ", Num_halos

    return Mbin_edge, Mbin_cent, Mcounts, Rshell, count_shell, Pshell_hist, stdshell_hist
