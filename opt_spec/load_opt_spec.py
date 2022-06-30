#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:49:29 2021

@author: yuhanyao
"""
import numpy as np
import extinction
from copy import deepcopy
from scipy.interpolate import interp1d
import astropy.io.ascii as asci
import astropy.constants as const
from astropy.time import Time

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

import sys
sys.path.append("/Users/yuhanyao/Dropbox/Projects/AT2020mrf/code/helper")
from load_opt_phot import load_18cow_phot, planck_lambda
from specconvolve import convolve_with_constant_velocity_kernel



def crop_telluric(wave, flux, 
                  wv1_range = (6758, 6838),
                  wv2_range = (7470, 7585)):
    ix1 = (wave>wv1_range[0])&(wave<wv1_range[1])
    ix2 = (wave>wv2_range[0])&(wave<wv2_range[1])
    ix = ix1 | ix2
    wave = wave[~ix]
    flux = flux[~ix]
    return wave, flux


def bin_spec(v4, y4, binning = 1):
    if binning != 1:
        yy6 = deepcopy(y4)
        vv6 = deepcopy(v4)
        rest = len(yy6)%binning
        if rest!=0:
            vv6 = vv6[:(-1)*rest]
            yy6 = yy6[:(-1)*rest]
        nnew = int(len(yy6) / binning)
        yy6_new = yy6.reshape(nnew, binning)
        yy6_new = np.sum(yy6_new, axis=1)
        y4 = yy6_new / binning
        vv6_new = vv6.reshape(nnew, binning)
        vv6_new = np.sum(vv6_new, axis=1)
        v4 = vv6_new / binning
    yy4 = np.repeat(y4, 2, axis=0)
    v4diff = np.diff(v4)
    v4diff_left = np.hstack([v4diff[0], v4diff])
    v4diff_right = np.hstack([v4diff, v4diff[-1]])
    vv4 = np.repeat(v4, 2, axis=0)
    vv4[::2] -= v4diff_left/2
    vv4[1::2] += v4diff_right/2
    return vv4, yy4


def load_mrf_sp(v_con = 2e+2):
    z = 0.1353
    t0 = 59012
    ebv = 0.018
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D/10)
    dis_mod_ = dis_mod - 2.5 * np.log10(1+z)

    spec = asci.read("../data/data_20mrf/spec/tns_2020mrf_2020-06-17_10-40-15_FTN_FLOYDS-N_Global_SN_Project.ascii")
    tspec = Time("2020-06-17T10:40:15").mjd
    phase = (tspec - t0) / (1+z)
    
    dt = {}
    
    wave = spec["col1"].data / (1+z)
    flux = spec["col2"].data
    
    tau =  extinction.ccm89(spec["col1"].data, 3.1*ebv, 3.1) / 1.086
    flux0 = flux * np.exp(tau)
    
    # g-band extinction corrected AB magnitude = 19.02 mag
    lambda_eff_g = 4813.966872932173
    lamb_norm = lambda_eff_g/(1+z)
    nu_eff_g = const.c.cgs.value / (lambda_eff_g * 1e-8)
    mobs0_g = 19.01085007
    M0_g = mobs0_g - dis_mod_
    
    #  erg/AA/s
    Llamb = 3631e-23 * 10**(-0.4 * M0_g) * nu_eff_g / lambda_eff_g * 4 * np.pi * (10 * const.pc.cgs.value)**2

    func1 = interp1d(wave, flux0)
    scale1 = Llamb / func1(lamb_norm)
    flux0 *= scale1
    wave_con, flux0_con = convolve_with_constant_velocity_kernel(wave, flux0, v_con)
    ix = (wave_con>(wave[0]+20))&(wave_con<(wave[-1]-20))
    wave_con = wave_con[ix]
    flux0_con = flux0_con[ix]
    
    dt["sp"] = {"wave": wave, "phase": phase, 
                 "wave_con": wave_con, 
                 "flux0": flux0, "flux0_con": flux0_con}
    return dt
    


def load_cow_sps(lamb_norm = 6000, v_con = 2e+2):
    """
    For AT2018cow
    t0 = 58285 # 
    lamb_norm = 6000 # rest-frame wavelength
    """
    tbp = load_18cow_phot()
    funcT = interp1d(tbp["phase_rest"], tbp["T"])
    funcR = interp1d(tbp["phase_rest"], tbp["R"]/const.R_sun.cgs.value)
    
    dt = {}
    
    z = 0.0141
    t0jd = 58285
    ebv = 0.07
    
    ###### LT spectrum at 4.95 days
    tb1 = asci.read("../data/data_cow/spectra/AT2018cow_20180620_LT_v2.ascii")
    f = open("../data/data_cow/spectra/AT2018cow_20180620_LT_v2.ascii")
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    wave1 = tb1["col1"].data/(1+z)
    flux1 = tb1["col2"].data
    
    tau =  extinction.ccm89(tb1["col1"].data, 3.1*ebv, 3.1) / 1.086
    flux10 = flux1 * np.exp(tau)
    
    wave1, flux10 = crop_telluric(wave1, flux10, 
                                 wv1_range = (6758, 6869), 
                                 wv2_range = (7474, 7565))
    
    ind = np.array([x[:11]=="# MJD     =" for x in lines])
    myline = lines[ind][0]  
    mjd= float(myline[15:-30])
    phase1 = (mjd - t0jd)/1+z
    phase1 = 4.946
    
    
    Llamb1 = planck_lambda(funcT(phase1), funcR(phase1), lamb_norm) #  erg/AA/s
    func1 = interp1d(wave1, flux10)
    scale1 = Llamb1 / func1(lamb_norm)
    flux10 *= scale1
    wave1_con, flux10_con = convolve_with_constant_velocity_kernel(wave1, flux10, v_con)
    ix = (wave1_con>(wave1[0]+20))&(wave1_con<(wave1[-1]-20))
    wave1_con = wave1_con[ix]
    flux10_con = flux10_con[ix]
    
    ###### HCT spectrum at 4.65 days
    tb2 = asci.read("../data/data_cow/spectra/AT2018cow_20180620_HCT_v1.ascii")
    wave2 = tb2["col1"].data/(1+z)
    flux2 = tb2["col2"].data
    phase2 = 4.651
    tau =  extinction.ccm89(tb2["col1"].data, 3.1*ebv, 3.1) / 1.086
    flux20 = flux2 * np.exp(tau)
    
    wave2, flux20 = crop_telluric(wave2, flux20, 
                                 wv1_range = (6757, 6802), 
                                 wv2_range = (7474, 7582))
    
    
    Llamb2 = planck_lambda(funcT(phase2), funcR(phase2), lamb_norm) #  erg/AA/s
    func2 = interp1d(wave2, flux20)
    scale2 = Llamb2 / func2(lamb_norm)
    flux20 *= scale2
    flux2
    wave2_con, flux20_con = convolve_with_constant_velocity_kernel(wave2, flux20, v_con)
    ix = (wave2_con>(wave2[0]+20))&(wave2_con<(wave2[-1]-20))
    wave2_con = wave2_con[ix]
    flux20_con = flux20_con[ix]
    

    
    ###### DSBP spectrum at 5.35 days
    tb3 = asci.read("../data/data_cow/spectra/AT2018cow_20180621_P200_v4.ascii")
    wave3 = tb3["col1"].data/(1+z)
    flux3 = tb3["col2"].data
    phase3 = 5.353
    tau =  extinction.ccm89(tb3["col1"].data, 3.1*ebv, 3.1) / 1.086
    flux30 = flux3 * np.exp(tau)
    
    wave3, flux30 = crop_telluric(wave3, flux30, 
                                 wv1_range = (6762, 6790), 
                                 wv2_range = (7478, 7532))
    
    
    Llamb3 = planck_lambda(funcT(phase3), funcR(phase3), lamb_norm) #  erg/AA/s
    func3 = interp1d(wave3, flux30)
    scale3 = Llamb3 / func3(lamb_norm)
    flux30 *= scale3
    wave3_con, flux30_con = convolve_with_constant_velocity_kernel(wave3, flux30, v_con)
    ix = (wave3_con>(wave3[0]+20))&(wave3_con<(wave3[-1]-20))
    wave3_con = wave3_con[ix]
    flux30_con = flux30_con[ix]
    
    ###### DCT spectrum at 7.145 days
    tb4= asci.read("../data/data_cow/spectra/AT2018cow_20180623_DCT_v1.ascii")
    wave4 = tb4["col1"].data/(1+z)
    flux4 = tb4["col2"].data
    phase4 = 7.145
    tau =  extinction.ccm89(tb4["col1"].data, 3.1*ebv, 3.1) / 1.086
    flux40 = flux4 * np.exp(tau)
    
    Llamb4 = planck_lambda(funcT(phase4), funcR(phase4), lamb_norm) #  erg/AA/s
    func4 = interp1d(wave4, flux40)
    scale4 = Llamb4 / func4(lamb_norm)
    flux40 *= scale4
    wave4_con, flux40_con = convolve_with_constant_velocity_kernel(wave4, flux40, v_con)
    ix = (wave4_con>(wave4[0]+20))&(wave4_con<(wave4[-1]-20))
    wave4_con = wave4_con[ix]
    flux40_con = flux40_con[ix]

    
    dt["sp1"] = {"wave": wave2, "phase": phase2, 
                 "wave_con": wave2_con, 
                 "flux0": flux20, "flux0_con": flux20_con}
    dt["sp2"] = {"wave": wave1, "phase": phase1,
                 "wave_con": wave1_con, 
                 "flux0": flux10, "flux0_con": flux10_con}
    dt["sp3"] = {"wave": wave3, "phase": phase3, 
                 "wave_con": wave3_con, 
                 "flux0": flux30, "flux0_con": flux30_con}
    dt["sp4"] = {"wave": wave4, "phase": phase4, 
                 "wave_con": wave4_con, 
                 "flux0": flux40, "flux0_con": flux40_con}
    return dt
    