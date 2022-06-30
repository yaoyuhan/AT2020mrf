#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:03:10 2021

@author: yuhanyao
"""
import numpy as np
import astropy.constants as const

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def Planck(nu=None, T=None):
    """
    >> I = Planck(nu=1e14, T=1e4)

    return black body intensity (power per unit area per solid angle per frequency)
    """
    h = const.h.cgs.value
    c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    x = np.exp(h*nu/(k_B*T))
    x = np.array(x)
    return 2*h/c**2 * nu**3 / (x-1)


def flux2lum(S_uJy, nu, z):
    """
    S_uJy: the observed flux density (observer's frame)
    nu: the effective frequency in the observers frame
    """
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    S_cgs = S_uJy * 1e-6 * 1e-23
    nuLnu = 4*np.pi * D_cm**2 * S_cgs * nu
    return nuLnu


def lum2flux(nuLnu, nu, z):
    """
    nu: the effective frequency in the TDE's frame
    """
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    S_cgs = nuLnu / (nu * 4*np.pi * D_cm**2) 
    S_uJy = S_cgs * 1e23 * 1e+6
    return S_uJy


# here we keep the function to correct to a single frequency
def cc_func(T, nu):
    """
    Color correction for blackbody spectrum
    (factor to multiply) from nu_kcL_{nu_kc} to nuL_nu
    
    nu_g = 3e10/4770e-8 # sdss g-band is 4720, ps1 is 4870, ZTF is 4810
    nu_kc = nu_g # this is the reference frequency
    """
    nu_kc=6.3e+14
    h_over_k =  6.6261e-27/1.3806e-16 # = const.h.cgs.value / const.k_B.cgs.value
    from_kc = (np.exp(h_over_k * nu_kc / T)-1.) / (np.exp(h_over_k * nu / T)-1.) * (nu/nu_kc)**4 
    return from_kc


def cc_bol(T, nu):
    """
    Color correction for blackbody spectrum
    (factor to multiply) from bolometric luminosity to nuLnu 
    
    note that nu is the frequency in TDE's rest-frame
    
    nuLnu / L_bol = nu pi B_nu(T) / (sigmaT^4)
    """
    return Planck(nu, T)*nu * np.pi /  (const.sigma_sb.cgs.value * T**4)


def T_evo(x, p_T, xpeak = 0, T_time=None):
    """
    Evolution of temperature
    """
    # if we get two parameters we do linear temperature change
    if len(p_T)==2:
        T0, dT = p_T[0], p_T[1]
        # assume no temperature evolition before maximum light
        # also assume no temperature evolution 7 days after max
        x_off = np.clip(x-xpeak, 0, 7)
        Ts = np.clip(T0 +  dT * x_off, 1e3, 1e5)
    # with more we do a linear interpolation on a grid
    else:
        Ts = 10**np.interp(x, T_time, p_T)
    return Ts
    

def get_cc(p, x, nu, xpeak=0, model_name='', T_time=None):
    """
    helper function to get color correction for different models
    model curve * cc = observed data
        for example, model curve is rest-frame g-band light curve
    
    Parameters:
        p: model fitting parameters
        x: time, rest-frame days relative to (crude estimation of) light curve peak
        xpeak: time of maximum light (model parameter)
        nu: frequency (filter effective frequency in object's rest-frame)
        model_name: can be chosen from 
            (1) nu_kc: both is the light curve in reference frequency nu_kc
            (2) bolo: both is the light curve in bolometric luminosity
            (3) Tflex: both is the light curve in bolometric luminosity
    """

    # grid with fixed points 
    if model_name == "Tflex":
        p_T = p[5:]                 
        """
        # exception for exponential model one less free par
        if 'exp' in model_name:
            p_T = p[4:] # T, dT
    
        # exception for model woth PL+const one more free par
        elif 'disk' in model_name:
            p_T = p[6:]   
        """
    # temperature + linear evolution        
    else:
        p_T = [10**p[-2], p[-1]] # T, dT 
        
    # model is nuLnu at nu_kc
    if model_name=='nu_kc':
    
        T_pl= T_evo(x, p_T, xpeak) # temperature as a function of time
        # (factor to multiply) from nu_kcL_{nu_kc} to nuL_nu
        cc = cc_func(T_pl, nu) 
    
    # model is BB curve    
    elif ('flex' in model_name) or ('bolo' in model_name):
        
        T_pl = T_evo(x, p_T, xpeak, T_time) 
        # (factor to multiply) from bolometric luminosity to nuLnu 
        cc = cc_bol(T_pl, nu)
    
    return cc
    
    
def L_flex(x, p_L, L_time):
    """
    flexible luminosity
    Parameters:
        x: at which to evaluate luminosity by interpolation
        p_L: parameters of luminosity (model)
        L_time: time of Luminosity (model)
    """
    return 10**np.interp(x, L_time, p_L)


def gauss(x, sigma):
    out = np.exp( -0.5 * x**2 / (sigma**2) )
    return out


def plrise_exp(p, x, nu, model_name='nu_kc', T_time=None):
    """
    power-law rise and exponential decay
    """
    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # luminosity at peak (this can be bolometric or L(nu_kc) depending on the setting)
    x_fl = p[2] # first light epoch
    b2 = 10**(p[3]) # decay rate
    n = p[4] # rise power-law index
    
    # power-law rise
    leftside = a1 * (x - x_fl)**n / (x_peak - x_fl)**n
    
    # exponential decay
    rightside = a1 * np.exp(-(x-x_peak)/b2) 
    
    # combine 
    leftside[x<=x_fl] = 0
    leftside[x>x_peak] = 0.
    rightside[x<=x_peak] = 0. 
    both = leftside + rightside

    # conversion from model curve to nuLnu of the data
    cc = get_cc(p, x, nu, xpeak=x_peak, model_name=model_name, T_time=T_time) 
    result = both * cc
    
    return result


def plrise_PL(p, x, nu, model_name='nu_kc', T_time=None):
    """
    power-law rise and power-law decay
    """
    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # luminosity at peak (this can be bolometric or L(nu_kc) depending on the setting)
    x_fl = p[2] # first light epoch
    n = p[3] # rise power-law index
    t0 = 10**p[4] # index
    alpha = p[5] # index
    
    # power-law rise
    leftside = a1 * (x - x_fl)**n / (x_peak - x_fl)**n
    
    # power-law decay
    rightside = a1 * ((x-x_peak+t0)/t0)**alpha
    
    # combine 
    leftside[x<=x_fl] = 0
    leftside[x>x_peak] = 0.
    rightside[x<=x_peak] = 0. 
    both = leftside + rightside

    # conversion from model curve to nuLnu of the data
    cc = get_cc(p, x, nu, xpeak=x_peak, model_name=model_name, T_time=T_time) 
    result = both * cc
    
    return result


def gauss_exp(p, x, nu, model_name='nu_kc', T_time=None): 
    """
    Gaussian rise and exponential decay
    Parameters:
        p: model fitting parameters
            p[0]: time of peak 
            p[1]: lg10(L) at peak, here luminosity can be 
                    bolometric if model_name == "bolo"
                    nu_{kc}L(nu_kc) if model_name == "nu_kc"
            p[2]: lg10(trise), here trise is the Gausisian rise timescale
            p[3]: lg10(tdecay), here tdecay is the exponential decay timescale
            p[4]: lg10(T), here T is the temperature in Kelvin
            p[5]: dT, linear evolution of temperature, in Kelvin/day
        x: time, rest-frame days relative to crude estimation of light curve peak
        nu: frequency (filter effective frequency redshifted to source)
        model_name: can be chosen from 
            (1) nu_kc: both is the light curve in reference frequency nu_kc
            (2) bolo: both is the light curve in bolometric luminosity
        T_time: set to None (not usd for this model)
    """
    
    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # luminosity at peak (this can be bolometric or L(nu_kc) depending on the setting)
    b1 = 10**(p[2]) # gaussian rise
    b2 = 10**(p[3]) # decay rate
    
    # gaussian rise
    leftside = a1*gauss(x-x_peak, b1)

    # exponential decay
    #a2 = a1 * gauss(0, b1)
    #rightside = a2 * np.exp(-(x-x_peak)/b2) 
    # note that gauss(0, b1)==1
    rightside = a1 * np.exp(-(x-x_peak)/b2) 
    
    # combine 
    leftside[x>x_peak] = 0.
    rightside[x<=x_peak] = 0. 
    both = leftside + rightside

    # conversion from model curve to nuLnu of the data
    cc = get_cc(p, x, nu, xpeak=x_peak, model_name=model_name, T_time=T_time) 
    result = both * cc
    
    return result


def gauss_PL(p, x, nu, model_name='bolo', T_time=None): 
    """
    Gaussian rise and power-law decay
    Parameters:
        p: model fitting parameters
            p[0]: time of peak 
            p[1]: lg10(L) at peak, here luminosity can be 
                    bolometric if model_name in ["bolo", "Tlex"]
                    nu_{kc}L(nu_kc) if model_name == "nu_kc"
            p[2]: lg10(trise), here trise is the Gausisian rise timescale
            p[3]: lg(t0), power-law normalization
            p[4]: alpha, power-law index
            if model_name in ["bolo", "nu_kc"]:
                p[5]: lg10(T), here T is the temperature in Kelvin
                p[6]: dT, linear evolution of temperature, in Kelvin/day
            elif model_name in ["Tlex"]:
                p[5]--p[5+N-1]: temperature in each of the N grids
            endif
        x: time, rest-frame days relative to crude estimation of light curve peak
        nu: frequency (filter effective frequency at the source's rest-frame)
        model_name: can be chosen from 
            (1) nu_kc: both is the light curve in reference frequency nu_kc
            (2) bolo: both is the light curve in bolometric luminosity
            (3) Tflex: both is the light curve in bolometric luminosity
        T_time: set to None unless model_name == Tflex
    """
    x_peak = p[0]   # time of peak
    a1 = 10**(p[1]) # luminosity/flux at peak
    b1 = 10**(p[2]) # gaussian rise
    
    # gaussian rise
    leftside = a1*gauss(x-x_peak, b1)

    # power-law decay
    a2 = a1 * gauss(0, b1)
    t0 = 10**p[3] # index
    alpha = p[4] # index
    rightside = a2 * ((x-x_peak+t0)/t0)**alpha
    
    # combine 
    leftside[x>x_peak] = 0.
    rightside[x<=x_peak] = 0. 
    both  = leftside + rightside
    
    # conversion from model curve to nuLnu of the data
    cc = get_cc(p, x, nu, xpeak=x_peak, model_name=model_name, T_time=T_time) 

    return both * cc


    
def gauss_flex(p, x, nu, model_name='bolo', T_time=None, L_time=None):
    """
    Gaussian rise and flexible decay 
    Parameters:
        p: model fitting parameters
        x: time
        nu: frequency (filter effective frequency redshifted to source)
        model_name:
            ???
        T_time:
            ???
        L_time:
            ???
    """
    x_peak = p[0]   # time of peak
    b1 = 10**(p[1])  # gaussian rise
    a1 = 10**p[2+len(T_time)+1]  # luminosity/flux at peak
    
    # select params for flexible temperature & luminosity evolution              
    p_L = p[2+len(T_time):2+len(T_time)+len(L_time)]

    # gaussian rise
    leftside = a1*gauss(x-x_peak, b1)
    
    # flexible decay
    rightside = L_flex(x, p_L, L_time=L_time)

    # combine
    leftside[x>x_peak] = 0.
    rightside[x<=x_peak] = 0. 
    both = leftside + rightside

    # conversion from model curve to nuLnu of the data
    cc = get_cc(p, x, nu, xpeak=x_peak, model_name=model_name, T_time=T_time)

    return both * cc    
    

def get_default_priordict(x, y, yerr, T_time, L_time):
    prior_dict_defaults = {}
    # time of peak --> checked
    prior_dict_defaults["tpeak"] =  {"min": -20, 
                                     "max": 20, 
                                     "sigma": None, 
                                     "value": 0}
    # lg(Lpeak) --> checked
    prior_dict_defaults["fpeak"] =  {"min": 42, 
                                     "max": 46,
                                     "sigma": None, 
                                     "value": max(np.log10(y[y>0]))}
    # lg(trise): Gaussian rise timescale sigma --> checked
    prior_dict_defaults["rise"] =   {"min": 0.4, # sigma = 2.51 days --> t1/2 = 1.317 sigma = 3.31 days
                                     "max": 2.0,  # if 1.5 sigma = 31.6 days --> t1/2 = 1.317 sigma = 41.6 days
                                     "sigma": None, 
                                     "value": 0.8}
    
    # lg(tdecay) -- only used when exponential decay --> checked
    prior_dict_defaults["decay"] =  {"min": 0.5, 
                                     "max": 3,
                                     "sigma": None,
                                     "value": 1}
    
    # lg(t0) -- only used when Model_name is in ["PL", "PL_bolo", ""]
    prior_dict_defaults["t0"] =     {"min": 0, 
                                     "max": 3,
                                     "sigma": None,
                                     "value": 1.5}
    
    # first-light epoch (used if the rise is power-law) -->  checked
    prior_dict_defaults["tfl"] =     {"min": -120, 
                                     "max": -21, 
                                     "sigma": None,
                                     "value": -30}
    # power-law rise index
    prior_dict_defaults["n"] =     {"min": 0.1, 
                                     "max": 5, 
                                     "sigma": None,
                                     "value": 0.8}
    
    # power law decay index
    prior_dict_defaults["alpha"] =  {"min": -8,
                                     "max": 0,
                                     "sigma": None,
                                     "defaul": -1}
    # lg(T), temperature at peak --> checked
    prior_dict_defaults["T"] =      {"min": 3.8, 
                                     "max": 5,
                                     "sigma": None,
                                     "value": 4.5}
    # tmperature evolution, Kelvin/day
    prior_dict_defaults["dT"] =     {"min": -800,
                                     "max": +800,
                                     "sigma": None,
                                     "value": 0}
    # white noise factor
    # added error = abs(y) * np.exp(inf)
    prior_dict_defaults["lnf"] =    {"min": -6, # 0.25% of abs(y)
                                     "max": -1.8, # 16.53% of abs(y)
                                     "sigma": None,
                                     "value": -4} # 1.8% of abs(y)
    # systematic uncertainties
    prior_dict_defaults["lgsigma0"] = {"min": np.log10(max(y[y>0]))-5, # 
                                       "max": np.log10(max(y[y>0]))-1,  # 
                                       "sigma": None, #
                                       "value": np.log10(max(y[y>0]))-2}
    # flexible evolution of lg(T)
    for T in T_time:
        prior_dict_defaults["T"+str(int(T))] =  {"min": 3,
                                                 "max": 6,
                                                 "sigma": None,
                                                 "value": 4.5}
    # flexible evolution of lg(L)
    for L in L_time:
        prior_dict_defaults["L"+str(int( L))] =  {"min": np.log10(min(y[y>0]))-2,
                                                  "max": np.log10(max(y[y>0]))+1,
                                                  "sigma": None, 
                                                  "value": max(np.log10(y[y>0]))}
    return prior_dict_defaults
    