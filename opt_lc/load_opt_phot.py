#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:23:45 2021

@author: yuhanyao
"""
import extinction
import numpy as np
import pandas as pd
import astropy.io.ascii as asci
from astropy.io import fits
from astropy.table import Table
from copy import deepcopy
import astropy.constants as const

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def planck_lambda(T, Rbb, lamb):
    '''
    T in the unit of K
    Rbb in the unit of Rsun
    lamb in the unit of AA
    '''
    ANGSTROM = 1.0e-8
    # convert to cm for planck equation
    lamb2 = lamb * ANGSTROM
    x = const.h.cgs.value * const.c.cgs.value / (const.k_B.cgs.value * T * lamb2)
    x = np.array(x)
    Blambda = (2. * const.h.cgs.value * const.c.cgs.value**2 ) /  (lamb2**5. ) / (np.exp(x) - 1. )
    # convert back to ANGSTROM   
    spec = Blambda*ANGSTROM # in units of erg/cm2/Ang/sr/s
    Rbb *= const.R_sun.cgs.value
    spec1 = spec * (4. * np.pi * Rbb**2) * np.pi # erg/AA/s
    # spec1 *= 1./ (4*np.pi*D**2) to correct for distance
    return spec1


def load_18cow_phot():
    filename = "../data/data_cow/phot/Perley2019_tab4.dat"
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    #colnames = lines[0].split("\t")
    lines = lines[1:-1]
    
    mjds = []
    Lbbs = []
    Lbbs_unc_left = []
    Lbbs_unc_right = []
    Rbbs = []
    Rbbs_unc_left = []
    Rbbs_unc_right = []
    Tbbs = []
    Tbbs_unc_left = []
    Tbbs_unc_right = []
    for i in range(len(lines)):
        mylinesub = lines[i].split("\t")
        tstr = mylinesub[0]
        Lstr = mylinesub[1]
        Rstr = mylinesub[2]
        Tstr = mylinesub[3][:-2]
        mjds.append(float(tstr))
        Lstr1 = Lstr[:10].replace(" ", "")
        Lstr2 = Lstr[10:19]
        Lstr3 = Lstr[20:].replace(" ", "")
        
        Lbbs.append(float(Lstr1))
        Lbbs_unc_right.append(float(Lstr2))
        Lbbs_unc_left.append(float(Lstr3))
        
        Rbbs.append(float(Rstr.split("+")[0]))
        Rbbs_unc_right.append(float(Rstr.split("+")[-1].split("−")[0]))
        Rbbs_unc_left.append(float(Rstr.split("+")[-1].split("−")[-1]))
        
        Tbbs.append(float(Tstr.split("+")[0]))
        Tbbs_unc_right.append(float(Tstr.split("+")[-1].split("−")[0]))
        Tbbs_unc_left.append(float(Tstr.split("+")[-1].split("−")[-1]))
    
    Lbbs = np.array(Lbbs)* const.L_sun.cgs.value
    Lbbs_unc_right = np.array(Lbbs_unc_right)* const.L_sun.cgs.value
    Lbbs_unc_left = np.array(Lbbs_unc_left)* const.L_sun.cgs.value
    Rbbs = np.array(Rbbs) * const.au.cgs.value
    Rbbs_unc_right = np.array(Rbbs_unc_right) * const.au.cgs.value
    Rbbs_unc_left = np.array(Rbbs_unc_left) * const.au.cgs.value
    Tbbs = np.array(Tbbs)*1e+3
    Tbbs_unc_right = np.array(Tbbs_unc_right)*1e+3
    Tbbs_unc_left = np.array(Tbbs_unc_left)*1e+3
        
    tb = Table(data = [mjds, 
                  Lbbs, Lbbs_unc_right, Lbbs_unc_left,
                  Rbbs, Rbbs_unc_right, Rbbs_unc_left,
                  Tbbs, Tbbs_unc_right, Tbbs_unc_left],
          names = ["mjd", 
                   "L", "L_unc_right", "L_unc_left",
                   "R", "R_unc_right", "R_unc_left",
                   "T", "T_unc_right", "T_unc_left"])
    z = 0.0140
    t0 = 58285
    tb["phase_rest"] = (tb["mjd"] - t0) / (1+z)
    return tb


def predict_18cow_mag(lambs = [4317]):
    """
    Predict the absolute magnitude of AT2018cow
    lambs: array of rest-frame wavelength in AA
    """
    tb = load_18cow_phot()
    L = tb["L"].data
    R = tb["R"].data
    
    T = tb["T"].data
    t = tb["phase_rest"].data
    absmags = []
    
    for i in range(len(lambs)):
        lamb = lambs[i]
    
        Llamb = planck_lambda(T, R/const.R_sun.cgs.value, lamb)
        L = lamb * Llamb  # erg/s
        nu = const.c.cgs.value / (lamb*1e-8) # Hz
        Lnu = L/nu # erg/s/Hz
        fnu_10pc = Lnu / (4 * np.pi * (10*const.pc.cgs.value)**2 ) # erg/s/Hz/cm^2
        absmag = -2.5 * np.log10(fnu_10pc / 3631e-23)
        absmags.append(absmag)
    return t, absmags


def load_21csp_phot():
    filename = "../data/data_21csp/bbpar.txt"
    df = asci.read(filename)
    
    mjds = df["col1"].data
    Lbbs_ = []
    Lbbs_unc_left = []
    Lbbs_unc_right = []
    Rbbs_ = []
    Rbbs_unc_left = []
    Rbbs_unc_right = []
    Tbbs_ = []
    Tbbs_unc_left = []
    Tbbs_unc_right = []
    for i in range(len(df)):
        Lstr = df["col3"][i]
        Rstr = df["col4"][i]
        Tstr = df["col5"][i]
        
        Lstr1 = Lstr[:5]
        Lstr2 = Lstr[7:11]
        Lstr3 = Lstr[13:]
        
        Lbbs_.append(float(Lstr1))
        Lbbs_unc_right.append(float(Lstr2))
        Lbbs_unc_left.append(float(Lstr3))
        
        Rbbs_.append(float(Rstr.split("+")[0]))
        Rbbs_unc_right.append(float(Rstr.split("+")[-1].split("-")[0]))
        Rbbs_unc_left.append(float(Rstr.split("+")[-1].split("-")[-1]))
        
        Tbbs_.append(float(Tstr.split("+")[0]))
        Tbbs_unc_right.append(float(Tstr.split("+")[-1].split("-")[0]))
        Tbbs_unc_left.append(float(Tstr.split("+")[-1].split("-")[-1]))
    
    Lbbs = 10**np.array(Lbbs_)#* const.L_sun.cgs.value
    Lbbs_right = 10**(np.array(Lbbs_) + np.array(Lbbs_unc_right)) 
    Lbbs_left = 10**(np.array(Lbbs_) - np.array(Lbbs_unc_left)) 
    Lbbs_unc_right = Lbbs_right - Lbbs
    Lbbs_unc_left = Lbbs - Lbbs_left
    
    Rbbs = 10**np.array(Rbbs_) 
    Rbbs_right = 10**(np.array(Rbbs_) + np.array(Rbbs_unc_right)) 
    Rbbs_left = 10**(np.array(Rbbs_) - np.array(Rbbs_unc_left)) 
    Rbbs_unc_right = Rbbs_right - Rbbs
    Rbbs_unc_left = Rbbs - Rbbs_left
    
    Tbbs = 10**np.array(Tbbs_)
    Tbbs_right = 10**(np.array(Tbbs_) + np.array(Tbbs_unc_right)) 
    Tbbs_left = 10**(np.array(Tbbs_) - np.array(Tbbs_unc_left)) 
    Tbbs_unc_right = Tbbs_right - Tbbs
    Tbbs_unc_left = Tbbs - Tbbs_left
        
    tb = Table(data = [mjds, 
                  Lbbs, Lbbs_unc_right, Lbbs_unc_left,
                  Rbbs, Rbbs_unc_right, Rbbs_unc_left,
                  Tbbs, Tbbs_unc_right, Tbbs_unc_left],
          names = ["mjd", 
                   "L", "L_unc_right", "L_unc_left",
                   "R", "R_unc_right", "R_unc_left",
                   "T", "T_unc_right", "T_unc_left"])
    z = 0.084
    t0 = 59254.5
    tb["phase_rest"] = (tb["mjd"] - t0) / (1+z)
    return tb


def predict_21csp_mag(lambs = [4317]):
    """
    Predict the absolute magnitude of AT2018cow
    lambs: array of rest-frame wavelength in AA
    """
    tb = load_21csp_phot()
    L = tb["L"].data
    R = tb["R"].data
    
    T = tb["T"].data
    t = tb["phase_rest"].data
    absmags = []
    
    for i in range(len(lambs)):
        lamb = lambs[i]
    
        Llamb = planck_lambda(T, R/const.R_sun.cgs.value, lamb)
        L = lamb * Llamb  # erg/s
        nu = const.c.cgs.value / (lamb*1e-8) # Hz
        Lnu = L/nu # erg/s/Hz
        fnu_10pc = Lnu / (4 * np.pi * (10*const.pc.cgs.value)**2 ) # erg/s/Hz/cm^2
        absmag = -2.5 * np.log10(fnu_10pc / 3631e-23)
        absmags.append(absmag)
    return t, absmags


def bin_lc(tb, ycol = "uJy", eycol = "duJy"):
    """
    bin by day
    """
    mjd = tb["mjd"].values
    uJy = tb[ycol].values
    duJy = tb[eycol].values
    # bin size is in day
    mjd1 = int(np.floor(min(mjd)))
    mjd2 = int(np.ceil(max(mjd)))

    xbound = np.arange(mjd1, mjd2+1)
    nbin = len(xbound)-1
    xs = np.ones(nbin)
    ys = np.ones(nbin)
    eys  = np.ones(nbin)
    for j in range(nbin):
        xmin = xbound[j]
        xmax = xbound[j+1]
        ix = (mjd>=xmin)&(mjd<xmax)
        if np.sum(ix)!=0:
            xx = mjd[ix]
            data = uJy[ix]
            edata = duJy[ix]
            wdata = 1./edata**2
            y = np.sum(data*wdata)/np.sum(wdata)
            ey = 1 / np.sqrt(np.sum(wdata))
            ys[j] = y
            eys[j] = ey
            xs[j] = np.median(xx)
    ix = xs!=1
    xs = xs[ix]
    ys = ys[ix]
    eys = eys[ix]
    tb_new = Table(data = [xs, ys, eys], names = ["mjd", ycol, eycol])
    return tb_new.to_pandas()


def load_ztf_lc(dobin = True, mjd_max = 59060):
    myfile = "../data/data_20mrf/phot/forcedphotometry_req00012244_lc.txt"
    f = open(myfile)
    lines = f.readlines()
    f.close()
    tb = asci.read(lines[57:])
    colnames = (lines[55][1:].split('\n'))[0].split(', ')
    for j in range(len(colnames)):
        tb.rename_column('col%d'%(j+1), colnames[j])   
    if tb['forcediffimfluxunc'].dtype in ['<U16', '<U17', '<U18', '<U19']:
        ix = tb['forcediffimfluxunc']=='null'
        tb = tb[~ix]
        tb['forcediffimfluxunc'] = np.array(tb['forcediffimfluxunc'], dtype=float)
    tb['forcediffimflux'] = np.array(tb['forcediffimflux'], dtype=float)
    tb = tb.to_pandas()
    
    tb.rename(columns={'forcediffimflux':'Fpsf',
                       'forcediffimfluxunc':'Fpsf_unc',
                       'zpdiff':'zp',
                       "field": "fieldid",
                       'zpmaginpsciunc':'ezp',
                       'jd':'jdobs',
                       'forcediffimchisq':'chi2_red',
                       'sciinpseeing':'seeing'}, inplace=True)
    
    F0 = 10**(tb['zp'].values/2.5)
    eF0 = F0 / 2.5 * np.log(10) * tb['ezp'].values
    Fpsf = tb['Fpsf'].values
    eFpsf = tb['Fpsf_unc'].values
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    tb['Fratio'] = Fratio
    tb['Fratio_unc'] = eFratio
    filt = tb['filter']
    filterid = np.zeros(len(tb))
    filterid[filt=='ZTF_g']=1
    filterid[filt=='ZTF_r']=2
    filterid[filt=='ZTF_i']=3
    tb['filterid'] = filterid
    tb["mjd"] = tb["jdobs"]-2400000.5
    tb = tb[tb.programid==1]
    tb = tb[tb["infobitssci"].values==0]
    tb = tb[tb["seeing"].values<4]
    tb = tb[tb["scisigpix"].values<20]
    tb = tb.drop(columns=["infobitssci", "procstatus", "chi2_red", "seeing", "scisigpix",
                         'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp',
                         "jdobs", 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap',
                         'forcediffimsnr', 'exptime', 'adpctdif1', 'adpctdif2', 'aperturecorr',
                         'clrcoeff', 'clrcoeffunc', "zpmaginpscirms", "rfid", "ncalmatches", 
                         "scibckgnd"])
    tb['fcqfid'] = tb['fieldid']*10000 + tb['ccdid']*100 + tb['qid']*10 + tb['filterid']
    tb = tb[tb["fcqfid"].values!=17601121]
    tb = tb[tb["fcqfid"].values!=17601122]
    
    tb = tb[tb.programid==1]
    tb = tb[tb.Fratio_unc<1e-8]
    tb = tb[tb.mjd>58980]
    tb = tb[tb.mjd<mjd_max]
    
    tb["uJy"] = tb["Fratio"]*3631e+6
    tb["duJy"] = tb["Fratio_unc"]*3631e+6
    
    if dobin:
        ixg = tb["filterid"].values==1
        ixr = tb["filterid"].values==2
        tbg = bin_lc(tb[ixg])
        tbg["F"] = "g"
        tbr = bin_lc(tb[ixr])
        tbr["F"] = "r"
        tb = pd.concat([tbg, tbr])
        tb = tb.sort_values(by="mjd")
        tb["Fratio"] = tb["uJy"].values / (3631e+6)
        tb["Fratio_unc"] = tb["duJy"].values / (3631e+6)    
    return tb


def load_atlas_lc(dobin = True, mjd_max = 59060):
    file_atlas = "../data/data_20mrf/phot/job65902.txt"
    lca = asci.read(file_atlas)
    tb_atlas = lca.to_pandas()
    #tb_atlas = tb_atlas[tb_atlas.duJy<50]
    #tb_atlas = tb_atlas[tb_atlas["chi/N"].values<5]
    tb = tb_atlas.drop(columns=["err", 'RA', 'Dec', 'x', 'y', 'mag5sig', 'Sky', 'Obs'])
    tb.rename(columns={'##MJD':'mjd'}, inplace=True)
    
    tb = tb.sort_values(by="mjd")
    tb["Fratio"] = tb["uJy"].values / (3631e+6)
    tb["Fratio_unc"] = tb["duJy"].values / (3631e+6)    
    tb = tb[tb.Fratio_unc<1e-8]
    tb = tb[tb.mjd>58980]
    tb = tb[tb.mjd<mjd_max]
    tb = tb[tb["chi/N"].values<2]
    tb = tb.drop(columns=["chi/N", "m", "dm", "maj", "min", "phi", "apfit"])
    
    if dobin:
        ixc = tb["F"].values=="c"
        ixo = tb["F"].values=="o"
        tbc = bin_lc(tb[ixc], ycol = "uJy", eycol = "duJy")
        tbc["F"] = "c"
        tbo = bin_lc(tb[ixo], ycol = "uJy", eycol = "duJy")
        tbo["F"] = "o"
        tb = pd.concat([tbc, tbo])
    tb = tb.sort_values(by="mjd")
    tb["Fratio"] = tb["uJy"].values / (3631e+6)
    tb["Fratio_unc"] = tb["duJy"].values / (3631e+6)    
    return tb


def deredden_df(tb, ebv):
    """
    perform extinction correction
    """
    if 'mag' in tb.columns:
        tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1) # extinction in magnitude
    if "limmag" in tb.columns:
        tb['limmag0'] = tb["limmag"] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1) # extinction in magnitude
    return tb


def load_koala_lc(SNT = 3):
    myfile = "../data/data_koala/forcedphotometry_req00013425_lc.txt"
    f = open(myfile)
    lines = f.readlines()
    f.close()
    tb = asci.read(lines[57:])
    colnames = (lines[55][1:].split('\n'))[0].split(', ')
    for j in range(len(colnames)):
        tb.rename_column('col%d'%(j+1), colnames[j])   
    if tb['forcediffimfluxunc'].dtype in ['<U16', '<U17', '<U18', '<U19']:
        ix = tb['forcediffimfluxunc']=='null'
        tb = tb[~ix]
        tb['forcediffimfluxunc'] = np.array(tb['forcediffimfluxunc'], dtype=float)
    tb['forcediffimflux'] = np.array(tb['forcediffimflux'], dtype=float)
    tb = tb.to_pandas()
    
    tb.rename(columns={'forcediffimflux':'Fpsf',
                       'forcediffimfluxunc':'Fpsf_unc',
                       'zpdiff':'zp',
                       "field": "fieldid",
                       'zpmaginpsciunc':'ezp',
                       'jd':'jdobs',
                       'forcediffimchisq':'chi2_red',
                       'sciinpseeing':'seeing'}, inplace=True)
    
    F0 = 10**(tb['zp'].values/2.5)
    eF0 = F0 / 2.5 * np.log(10) * tb['ezp'].values
    Fpsf = tb['Fpsf'].values
    eFpsf = tb['Fpsf_unc'].values
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    tb['Fratio'] = Fratio
    tb['Fratio_unc'] = eFratio
    filt = tb['filter']
    filterid = np.zeros(len(tb))
    filterid[filt=='ZTF_g']=1
    filterid[filt=='ZTF_r']=2
    filterid[filt=='ZTF_i']=3
    tb['filterid'] = filterid
    tb["mjd"] = tb["jdobs"]-2400000.5
    tb = tb[tb["infobitssci"].values==0]
    tb = tb[tb["seeing"].values<4]
    tb = tb[tb["scisigpix"].values<20]
    tb = tb.drop(columns=["infobitssci", "procstatus", "chi2_red", "seeing", "scisigpix",
                         'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp',
                         "jdobs", 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap',
                         'forcediffimsnr', 'exptime', 'adpctdif1', 'adpctdif2', 'aperturecorr',
                         'clrcoeff', 'clrcoeffunc', "zpmaginpscirms", "rfid", "ncalmatches", 
                         "scibckgnd"])
    tb['fcqfid'] = tb['fieldid']*10000 + tb['ccdid']*100 + tb['qid']*10 + tb['filterid']
    tb = tb[tb.mjd<58400]
    tb = tb[tb.Fratio_unc<1e-8]
    tb = tb[tb.mjd>58360]
    
    tb["uJy"] = tb["Fratio"]*3631e+6
    tb["duJy"] = tb["Fratio_unc"]*3631e+6
    dobin = True
    if dobin == True:
        ixg = tb["filterid"].values==1
        ixr = tb["filterid"].values==2
        tbg = bin_lc(tb[ixg])
        tbg["F"] = "g"
        tbr = bin_lc(tb[ixr])
        tbr["F"] = "r"
        tb = pd.concat([tbg, tbr])
        tb = tb.sort_values(by="mjd")
        tb["Fratio"] = tb["uJy"].values / (3631e+6)
        tb["Fratio_unc"] = tb["duJy"].values / (3631e+6)   
    z = 0.2714
    t0 = 58372.4206 #2458372.9206-2400000.5
    tb["survey"] = "ZTF"
    
    tb["phase_obs"] = tb["mjd"] - t0
    tb["phase_rest"] = (tb["mjd"] - t0)/(1+z)
    
    lambda_eff_g = 4813.966872932173
    lambda_eff_r = 6421.811631761602
    
    ixg = tb['F'].values=="g"
    ixr = tb['F'].values=="r"
    
    waves = np.zeros(len(tb))
    waves[ixg] = lambda_eff_g
    waves[ixr] = lambda_eff_r
    
    tb["wave"] = waves
    Fratio = tb["Fratio"].values
    Fratio_unc = tb["Fratio_unc"].values
    ix = Fratio > SNT*Fratio_unc
    
    det = deepcopy(tb[ix])
    nondet = deepcopy(tb[~ix])

    mags = -2.5 * np.log10(Fratio[ix])
    emags = 2.5 / np.log(10) * Fratio_unc[ix] / Fratio[ix]
    det["mag"] = mags
    det["mag_unc"] = emags

    det = det.sort_values(by="mjd")
    nondet = nondet.sort_values(by="mjd")
    nondet["limmag"] = -2.5 * np.log10(3 * Fratio_unc[~ix])
    
    ebv = 0.0439 # S & F (2011), max = 0.0182, min = 0.0158	0., ref pixel 0.0181	0.0
    nondet = deredden_df(nondet, ebv)
    det = deredden_df(det, ebv)
    
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D/10)
    det["mag0_abs"]  = det["mag0"] - dis_mod
    nondet["limmag0_abs"]  = nondet["limmag0"] - dis_mod
    return det, nondet
    


def load_fox_lc(SNT = 3, mjd_max = 59060, t0=59012, z=0.115):
    tb_ztf = load_ztf_lc(dobin = True, mjd_max=mjd_max)
    tb_atlas = load_atlas_lc(dobin=True, mjd_max=mjd_max)
    tb_ztf["survey"] = "ZTF"
    tb_atlas["survey"] = "ATLAS"

    tb = pd.concat([tb_ztf, tb_atlas])
    tb["phase_obs"] = tb["mjd"] - t0
    tb["phase_rest"] = (tb["mjd"] - t0)/(1+z)
    
    lambda_eff_c = 5183.87
    lambda_eff_o = 6632.15
    lambda_eff_g = 4813.966872932173
    lambda_eff_r = 6421.811631761602
    #lambda_eff_i = 7883.058236798149
    
    ixg = tb['F'].values=="g"
    ixr = tb['F'].values=="r"
    ixc = tb["F"].values=="c"
    ixo = tb["F"].values=="o"
    
    waves = np.zeros(len(tb))
    waves[ixg] = lambda_eff_g
    waves[ixr] = lambda_eff_r
    waves[ixc] = lambda_eff_c
    waves[ixo] = lambda_eff_o
    
    tb["wave"] = waves
    Fratio = tb["Fratio"].values
    Fratio_unc = tb["Fratio_unc"].values
    ix = Fratio > SNT*Fratio_unc

    det = deepcopy(tb[ix])
    nondet = deepcopy(tb[~ix])

    mags = -2.5 * np.log10(Fratio[ix])
    emags = 2.5 / np.log(10) * Fratio_unc[ix] / Fratio[ix]
    det["mag"] = mags
    det["mag_unc"] = emags

    det = det.sort_values(by="mjd")
    nondet = nondet.sort_values(by="mjd")
    nondet["limmag"] = -2.5 * np.log10(3 * Fratio_unc[~ix])
    
    ebv = 0.018 # S & F (2011), max = 0.0182, min = 0.0158	0., ref pixel 0.0181	0.0
    nondet = deredden_df(nondet, ebv)
    det = deredden_df(det, ebv)
    return det, nondet
    
      