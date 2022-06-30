from __future__ import print_function, division, absolute_import


import numpy as np
from astropy.constants import c


def convolve_with_constant_velocity_kernel(wave, flux, v = 1e4):
    """
    Convolve a spectrum with a gaussian kernel of constant velocity
    
    Parameters
    ----------
    wave : float arr
        Wavelength values for each pixel in the spectrum. Velocity 
        calculations assume wavelength is given in Ang.
    
    flux : float arr
        Flux values for each pixel in the spectrum. Relative accuracy
        is assumed, but absolute scale is not necessary.
    
    v : float, optional (default = 1e4)
        Full-width half-max velocity of the gaussian kernel. Velocity 
        must be in units of km/s.
    
    Returns
    -------
    interp_grid : float arr
        Wavelength (in same units as wave) at each pixel in the 
        interpolated grid used for the convolution
    
    conv_flux : float arr
        Flux at each pixel in wave following convolution with the 
        constant velocity width kernel
    """
    
    deltaAng = np.median(np.diff(wave))
    interp_grid = np.arange(min(wave), max(wave), deltaAng)
    interp_flux = np.interp(interp_grid, wave, flux)

    var_kern_fwhm = v*1e3/c.value*interp_grid # 1e3 converts c from m/s to km/s
    conv_flux = np.empty(len(interp_flux))

    for pix in range(len(conv_flux)):
        sigmaKern = var_kern_fwhm[pix]/(2*np.sqrt(2*np.log(2)))
        gx = np.arange(-4*sigmaKern, 4*sigmaKern, deltaAng)
        kern = deltaAng/np.sqrt(2*np.pi*sigmaKern**2)*np.exp(-1./2*(gx/sigmaKern)**2)
        gauss_flux = np.convolve(interp_flux, kern, mode = 'same')
        conv_flux[pix] = gauss_flux[pix]
    
    return interp_grid, conv_flux
