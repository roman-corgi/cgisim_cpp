#   Copyright 2026 California Institute of Technology
# ------------------------------------------------------------------

import proper
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import roman_preflight_proper
from roman_preflight_proper import trim

def run_spc_wide_band1():

    nlam = 7
    lam0 = 0.575
    bandwidth = 0.10
    minlam = lam0 * (1 - bandwidth/2)
    maxlam = lam0 * (1 + bandwidth/2)
    lam_array = np.linspace( minlam, maxlam, nlam )

    n = 512                 # output image dimension (must be power of 2)
    final_sampling = 0.1    # output sampling in lam0/D


    print( "Computing aberrated coronagraphic field using DM solution for flat WFE" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_flat_wfe_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_flat_wfe_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide_band1', 'use_errors':1, 'polaxis':10, \
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image_flat = np.sum( images, 0 ) / nlam

    print( "Computing aberrated coronagraphic field using best dark hole DM solution" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_5e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_5e-9_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide_band1', 'use_errors':1, 'polaxis':10, \
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image_best = np.sum( images, 0 ) / nlam

    print( "Computing aberrated coronagraphic field using mild dark hole DM solution" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_8e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_8e-9_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide_band1', 'use_errors':1, 'polaxis':10, \
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image_mild = np.sum( images, 0 ) / nlam

    print( "Computing aberrated coronagraphic field using worst dark hole DM solution" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_2e-8_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_band1_ni_2e-8_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide_band1', 'use_errors':1, 'polaxis':10, \
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image_worst = np.sum( images, 0 ) / nlam

    print( "Computing no-FPM source" )

    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
            PASSVALUE={'use_fpm':0, 'cor_type':'spc-wide_band1', 'use_errors':1, 'polaxis':10, \
            'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    psfs = np.abs(fields)**2
    psf = np.sum( psfs, 0 ) / nlam
    max_psf = np.max(psf)

    ni_flat = image_flat / max_psf
    ni_best = image_best / max_psf
    ni_mild = image_mild / max_psf
    ni_worst = image_worst / max_psf

    fig, ax  = plt.subplots( nrows=2, ncols=2, figsize=(10,10) )
    ax = ax.flatten()

    im = ax[0].imshow(ni_flat, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[0].add_patch(circ_in)
    ax[0].add_patch(circ_out)
    ax[0].set_title('Flattened WFE')
    fig.colorbar(im, ax=ax[0], shrink=0.5) 

    im = ax[1].imshow(ni_worst, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[1].add_patch(circ_in)
    ax[1].add_patch(circ_out)
    ax[1].set_title('2e-8 Dark hole')
    fig.colorbar(im, ax=ax[1], shrink=0.5) 

    im = ax[2].imshow(ni_mild, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[2].add_patch(circ_in)
    ax[2].add_patch(circ_out)
    ax[2].set_title('8e-9 Dark hole')
    fig.colorbar(im, ax=ax[2], shrink=0.5) 

    im = ax[3].imshow(ni_best, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[3].add_patch(circ_in)
    ax[3].add_patch(circ_out)
    ax[3].set_title('5e-9 Dark hole')
    fig.colorbar(im, ax=ax[3], shrink=0.5) 

    plt.show()
    
if __name__ == '__main__':
    run_spc_wide_band1()
