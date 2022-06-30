#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:00:18 2021

@author: yuhanyao
"""
import numpy as np
import scipy.integrate as integrate
import astropy.constants as const


Msun = const.M_sun.cgs.value


def get_int_A(x, y, s):
    r = integrate.quad(lambda z: 2*z*np.exp(-2*z*y+z**2), 0, x)
    int_A = r[0]
    return int_A
    
    
def get_int_B(x, y, s):
    r = integrate.quad(lambda z: 2*z*np.exp(-2*z*y+2*z*s+z**2), 0, x)
    int_B = r[0]
    return int_B


def model_arnett_modified(ts_, taum_ = 3, Mni_ = 0.05, t0_ = 30):
    '''
    Calculate the flux of a radioactivity powered SN at photospheric phase
    
    ts is in the unit of day
    taum_ is in the unit of day
    Mni_ is in the unit of Msun
    
    The euqation is from
    Valenti 2008 MNRAS 383 1485V, Appendix A
    '''
    ts = ts_ * 24*3600 #  in seconds
    Mni = Mni_ * Msun
    tau_m = taum_ * 24 * 3600.
    t0 = t0_ * 24 * 3600.
    
    epsilon_ni = 3.9e+10 # erg / s / g
    epsilon_co = 6.78e+9 # erg / s / g
    tau_ni = 8.8 * 24 * 3600 # s
    tau_co = 111.3 * 24 * 3600 # s
    
    Ls = np.zeros(len(ts))
    for i in range(len(ts)):
        t = ts[i]
        if t<=0:
            Ls[i] = 0
        else:
            x = t / tau_m
            y = tau_m / (2 * tau_ni)
            s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)
            
            int_A = get_int_A(x, y, s)
            int_B = get_int_B(x, y, s)
            
            L = Mni * np.exp(-x**2) * ( (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B )
            Ls[i] = L
    # plt.loglog(ts/24/3600, Ls)
    Ls_modified = np.zeros(len(Ls))
    ix = ts > 0
    Ls_modified[ix] = Ls[ix]* (1. - np.exp(-1*(t0/ts[ix])**2) )
    return Ls_modified
