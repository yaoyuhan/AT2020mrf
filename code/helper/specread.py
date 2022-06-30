#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:12:11 2020

@author: yuhanyao
"""
import os
import numpy as np
import extinction
import astropy.io.ascii as asci
from astropy.time import Time
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev, interp1d
from collections import OrderedDict as odict

import sys
sys.path.append("/Users/yuhanyao/Documents/ZTF18abjpmzf/playground/")
from helper.specconvolve import convolve_with_constant_velocity_kernel


import matplotlib
import matplotlib.pyplot as plt
fs= 14
matplotlib.rcParams['font.size']=fs


def starlinelist(z=0.0):
    H_list = np.array([#3734.369, 
              3750.151, 3770.633, 3797.909, 
              3835.397, 
              3889.064, 
                3970.075,  # similar to H
                4101.734, 4340.472, 4861.35, 6562.79]) * (1+z)
    
    NaI_list = np.array([5889.95095, 5895.92424])
    
    KI_list = np.array([7664.89913, 7698.96456])
    
    OIII_list = np.array([ 4363.21,  #this is not detected
                          4958.91, 5006.84]) * (1+z)
    
    OII_list = np.array([3726.04, 3728.80 # oii line in the 2p3 configuration
                         ]) * (1+z)
    
    OI_list = np.array([5577, 
                        6300, 6363, 
                        7774.17
                        ]) * (1+z)
    
    SIII_list = np.array([6312.06,
                          9068.6, 9530.6 # nebula lines in the 2p2/2p4 configuration
                          ]) * (1+z)
    
    SII_list = np.array([6716.44, 6730.815 # nebula lines in the 2p3 configuration
                          ]) * (1+z)
    
    NII_list = np.array([6548.05, 6583.45]) *(1+z)
    
    
    # He I: Fig 1 of http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1991ApJ...383..308L&defaultprint=YES&filetype=.pdf
    # Also Filippenko Section 4.2
    # 5016, line from Fremling  PTF12os and iPTF13bvn:
    HeI_list = np.array([4472, 
                         5016,
                         5875, 
                         6678, 
                         7065 # what is this line? 
                         ]) * (1+z)
    
    FeII_list = np.array([4924, 5018, 5169])* (1+z)
    
    FeII_list = np.array([2250.02,
                          2261.00,
                          2346.01, # 2344
                          2374.46, # 2374
                          2383.59, # 2382
                          2586.49,
                          2600.23
                          ])* (1+z)
    
    CIII_list = np.array([4647.42, 4650.25, 4651.47])
    
    NIII_list = np.array([4634.14, 4640.64])
    
    HeII_list = np.array([3202.947,
                          3203.112,
                          3203.165,
                          4685.682 # this is n1=3, n2=4, recombination line?
                         ]) * (1+z)
    
    CaII_list = np.array([3933.66, 3968.47,
                          #7291.47, 7323.89, # 3p6 4s to 3p6 3d 
                          8498.02, 8542.09, 8662.14
                         ])* (1+z)
    
    MgII_list = np.array([2795.528, 2802.704,
                        8222.924, 8233.194, 8234.639,
                        9218.248, 9244.266])* (1+z)
    
    MgI_list = np.array([2779.831, 2852.127])* (1+z)
    
    Mgb_list = np.array([5167.321, 5172.684, 5183.604])* (1+z)
    
    DIB_list = np.array([5780.48, # Hobbs2008
                         6283.83 # Hobbs2008
                        ]) * (1+z)
    
    NeIII_list = np.array([3868.71 ])* (1+z) # Thnaks Rahcel!
    ArIII_list = np.array([7135.8, 7751.06 ]) * (1+z)# Thnaks Rahcel!
    yaolinelist = odict([('H_list', H_list),
                         ('HeI_list', HeI_list),
                         ('HeII_list', HeII_list),
                         ('OIII_list', OIII_list),
                         ('OII_list', OII_list),
                         ('OI_list', OI_list),
                         ('CIII_list', CIII_list),
                         ('NIII_list', NIII_list),
                         ('SIII_list', SIII_list),
                         ('SII_list', SII_list),
                         ('NII_list', NII_list),
                         ('CaII_list', CaII_list),
                         ('Mgb_list', Mgb_list),
                         ('NeIII_list', NeIII_list),
                         ('KI_list', KI_list),
                         ("ArIII_list", ArIII_list),
                         ("FeII_list", FeII_list),
                         ("MgI_list", MgI_list),
                         ("MgII_list", MgII_list),
                         ("DIB_list", DIB_list),
                         ("NaI_list", NaI_list)])
    return yaolinelist



def truncate_spec(wave, flux, ax2):
    yaolist = starlinelist(z=0)
    ix_retain = np.ones(len(wave), dtype=bool)
    
    H_list = yaolist['H_list'] # yes
    for wv in H_list:
        ix = abs(wave - wv) < 50
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color='r', zorder=5)
            
    keys = list(yaolist.keys()) 
    mylist = []
    for key in keys:
        currentlist = yaolist[key]
        for x in currentlist:
            mylist.append(x)
    for wv in mylist:
        ix = abs(wave - wv) < 10
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color='r', zorder=5)
        
    wave = wave[ix_retain]
    flux = flux[ix_retain]
    return wave, flux


def gaus(x, a, A, x0, sigma):
    return a + A * np.exp(-(x-x0)**2/(2*sigma**2))    


def measure_flux(wave, flux_line, line_center = 4861.35,  sigma_guess = 4.11,
                 line_bound_width = 1., linename = "Hbeta", 
                 line_left = 30, line_right = 30, a_width = 1e-2,
                 emission_flag = 1, doplot = True):
    """
    wave, flux_line: continuum subtracted spectrum
    
    line_center: wavelength
    line_bound_width: allow the line center to vary by a_width
    
    sigma_guess: Gauss profile sigma (initial guess)
    line_left, line_right: truncate the input spectrum boundary
    
    a_width: floow value is within [-a_width, a_width]
    """
    multi = 1e+16
    
    ind_h2 = (wave > line_center-line_left) & (wave < line_center+line_right)
    wave_h2 = wave[ind_h2]
    flux_h2_line = flux_line[ind_h2]
    
    a_fixed = 0.
    if emission_flag == 1:
        A_guess = max(flux_h2_line)*multi - a_fixed
        bounds = ((a_fixed-a_width, 0.5*A_guess, line_center-line_bound_width, sigma_guess/4),
                  (a_fixed+a_width, 2*A_guess, line_center+line_bound_width, sigma_guess*4))
    else:
        A_guess = min(flux_h2_line)*multi - a_fixed
        bounds = ((a_fixed-a_width, 3*A_guess, line_center-line_bound_width, sigma_guess/4),
                  (a_fixed+a_width, 0.8*A_guess, line_center+line_bound_width, sigma_guess*4))
    popt1, pcov1 = curve_fit(gaus, wave_h2, flux_h2_line*multi, 
                             p0=[a_fixed, A_guess, line_center, sigma_guess],
                             bounds=bounds)
    print ("line width = %.2f AA"%(popt1[-1]))
    #print ("line center is %.2f +- %.2f"%(popt1[2], np.sqrt(pcov1[2,2])))
    
    new_width = popt1[-1] * 4 # four times the sigma
    wvnew = np.linspace(line_center-new_width, line_center+new_width, 300)
    flnew = gaus(wvnew, *popt1)
    flux_gaus = (flnew - popt1[0])/multi
    
    wave_diff_ = wvnew[1:] - wvnew[:-1]
    wave_diff = 0.5 * (np.hstack([ wave_diff_[0],  wave_diff_])+ \
                       np.hstack([  wave_diff_,  wave_diff_[-1]]))
    flux_Hbeta = np.sum(wave_diff * flux_gaus)
    
    # =========================================================================
    # Calculate Uncertainty:
    # L is the Cholesky decomposition (lower matrix) of pcov1
    L = np.linalg.cholesky(pcov1)
    # you should find np.dot(L, L.T) == pcov1
    NSAMPLES = 100
    N = len(popt1)
    zprep = np.zeros((NSAMPLES, N))
    for i in range(N):
        zprep[:, i] = np.random.normal(0,1,(NSAMPLES))
        
    fluxes = np.zeros(NSAMPLES)
    wcenters = np.zeros(NSAMPLES)
    for i in range(NSAMPLES):
        p1_ = popt1 + np.dot(L, zprep[i])
        _flnew = gaus(wvnew, *p1_)
        _flux_gaus = (_flnew - p1_[0])/multi
        fluxes[i] = np.sum(wave_diff * _flux_gaus)
        wcenters[i] = p1_[-1]
    wcenter_unc = (np.percentile(wcenters, 84.13)- np.percentile(wcenters, 15.87))/2
    flux_unc = (np.percentile(fluxes, 84.13)- np.percentile(fluxes, 15.87))/2
    print ("line center is %.2f +- %.2f"%(popt1[2], wcenter_unc))
    
    if doplot == True:
        plt.figure(figsize=(6, 4))
        ax1 = plt.subplot(111)
        ax1.plot(wave_h2, flux_h2_line, 'k', label = "Observed")
        ax1.set_xlabel(r'$\lambda$'+' ('+r'$\AA$'+')')
        ax1.set_ylabel(r'$f_{\lambda, \rm obs} - f_{\lambda, \rm cont}$'+' (erg'+r'$\cdot$'+'cm'+r'$^{-2}$'+
                       r'$\cdot$'+'s'+r'$^{-1}$'+r'$\cdot$'+r'$\AA^{-1}$'+')') 
    
        ax1.plot(wvnew, flnew/multi, 'r--', label="Gaussian Model")
        ax1.plot(wvnew, flux_gaus, 'b:')
        ax1.set_title(linename, fontsize = fs)
        ax1.legend(loc = "upper left", fontsize=fs-2, frameon=False,
                   fancybox = False)
        plt.tight_layout()
        #plt.savefig(linename+".pdf")
    
    print ("line flux of %s is: %.2f +- %.2f 1e-16"%(linename, flux_Hbeta*multi, flux_unc*multi))
    dt = {}
    dt["flux"] = flux_Hbeta
    dt["flux_unc"] = flux_unc
    dt["popt"] = popt1
    dt["pcov"] = pcov1
    return dt


def add_telluric_circle(ax, x, y, rx=10, ry = 0.1, ls=0.5):
    inc = np.linspace(0, 2*np.pi, 100)
    xs = x + rx * np.cos(inc)
    ys = y + ry * np.sin(inc)
    ax.plot(xs, ys, 'k-', linewidth = ls)
    ax.plot([x, x], [y+ry, y-ry], 'k-', linewidth = ls)
    ax.plot([x-rx, x+rx], [y, y], 'k-', linewidth = ls)


def pblines(ax, H_list, color='m', label=None, ls = ':',
            tb = -42, tu = -32, linewidth = 1, alpha = 1):
    for i in range(len(H_list)):
        wv = H_list[i]
        if i==0 and (label is not None):
            ax.plot([wv, wv], [tb, tu], linestyle = ls, 
                    color=color, zorder = 1, label=label, linewidth=linewidth, alpha=alpha)
        else:
            ax.plot([wv, wv], [tb, tu], linestyle = ls, 
                    color=color, zorder = 1, linewidth=linewidth, alpha=alpha)


def get_keck(z=0, date = "20190412_Keck1_v2", 
             vkernel = None, t0jd = 58583.2,
             ebv = 0.022, dirpath = "../data/spectra/"):# Keck spectrum
    myfile = os.path.join(dirpath, "ZTF18abjpmzf_"+date+".ascii")
    f = open(myfile)
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    #lines = lines[100:200]
    ind = np.array([x[:11]=="# MJD     =" for x in lines])
    myline = lines[ind] [0]  
    mjd= float(myline[15:-30])
    phase = mjd - t0jd

    tb= asci.read(myfile)
    # tb = tb[tb["col1"]>3190]
    dt = {}
    dt["phase"] = np.round(phase, 2)
    xx = tb['col1'].data/(1+z)
    yy = tb['col2'].data
    yy_sky = tb['col3'].data
    yy_flag = tb['col8'].data
    eyy = tb['col4'].data
    ind = ~np.isnan(yy)
    dt["wave"] = xx[ind]*(1+z)
    dt['wave_rest'] = xx[ind]
    dt['spec_obs'] = yy[ind]
    dt['sky_obs'] = yy_sky[ind]
    dt['flag_obs'] = yy_flag[ind]
    dt['spec_obs_unc'] = eyy[ind]
    Aextmag =  extinction.ccm89(dt['wave_rest'], 3.1*ebv, 3.1) # extinction in magnitudes
    tau =  Aextmag / 1.086
    dt['spec_obs0'] = dt['spec_obs'] * np.exp(tau)
    dt["ln_spec_obs"] = np.log(dt['spec_obs0'])
    
    if vkernel != None:
        ww, ff = convolve_with_constant_velocity_kernel(dt['wave_rest'], dt['spec_obs0'], vkernel)
        dt['wave_con'] = ww
        dt['spec_con'] = ff
        dt["ln_spec_con"] = np.log(ff)
    return dt


def get_nires(z=0, ebv = 0.022):
    myfile = "../data/spectra/20201223_TSPEC/ztf18abjpmzf_61_merged_ascii.dat"
    mydt = np.loadtxt(myfile).T
    dt = {}
    dt["wave"] = mydt[0]
    dt["spec_obs"] = mydt[1]
    dt['wave_rest'] = mydt[0]/(1+z)
    Aextmag =  extinction.ccm89(dt['wave_rest'], 3.1*ebv, 3.1) # extinction in magnitudes
    tau =  Aextmag / 1.086
    dt['spec_obs0'] = dt['spec_obs'] * np.exp(tau)
    dt["ln_spec_obs"] = np.log(dt['spec_obs0'])
    return dt


def get_p200(z=0.0213, date = "20200731", substr="",
             vkernel = 200, ebv = 0.022, wvmin = 3500, wvmax = 9000):
    myfile = "../data/spectra/DBSP/DBSP_"+date+"_AT2019wey"+substr+".txt"
    f = open(myfile)
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    tb= asci.read(myfile)
    ix = (tb["col1"]>wvmin)&(tb["col1"]<wvmax)
    tb = tb[ix]
    """
    t = Time(['2019-04-24T11:17:05'], format='isot', scale='utc')
    phase = t.mjd[0] - t0jd
    if date == "20200731_P200_v3":
        ix = tb["col1"]<8900
        tb = tb[ix]
        ix = tb["col1"]>3600
        tb = tb[ix]
        ix = np.any([tb["col1"]<5450, tb["col1"]>5700], axis=0)
        tb = tb[ix]
    """
    dt = {}
    #dt["phase"] = np.round(phase, 2)
    dt['wave'] = tb['col1'].data
    dt['wave_rest'] = tb['col1'].data/(1+z)
    dt['spec_obs'] = tb['col2'].data
    Aextmag =  extinction.ccm89(dt['wave_rest'], 3.1*ebv, 3.1) # extinction in magnitudes
    tau =  Aextmag / 1.086
    dt['spec_obs0'] = dt['spec_obs'] * np.exp(tau)
    dt["ln_spec_obs"] = np.log(dt['spec_obs0'])
    
    ww, ff = convolve_with_constant_velocity_kernel(dt['wave_rest'], dt['spec_obs0'], vkernel)
    dt['wave_con'] = ww
    dt['spec_con'] = ff
    dt["ln_spec_con"] = np.log(ff)
    return dt


def get_ltspec(z=0.0213, date = '0409', vkernel = 800, t0jd = 58583.2, ebv = 0.022):
    myfile = "../data/spectra/ZTF18abfcmjw_2019"+date+"_LT_v1.ascii"
    f = open(myfile)
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    
    ind = np.array([x[:11]=="# MJD     =" for x in lines])
    myline = lines[ind] [0]  
    mjd= float(myline[15:-30])
    phase = mjd - t0jd
    
    tb = asci.read(myfile)
    dt = {}
    dt["phase"] = np.round(phase, 2)
    dt['wave_rest'] = tb['col1'].data/(1+z)
    dt['spec_obs'] = tb['col2'].data*8e-17
    Aextmag =  extinction.ccm89(dt['wave_rest'], 3.1*ebv, 3.1) # extinction in magnitudes
    tau =  Aextmag / 1.086
    dt['spec_obs0'] = dt['spec_obs'] * np.exp(tau)
    dt["ln_spec_obs"] = np.log(dt['spec_obs0'])

    ww, ff = convolve_with_constant_velocity_kernel(dt['wave_rest'], dt['spec_obs0'], vkernel)
    dt['wave_con'] = ww
    dt['spec_con'] = ff
    dt["ln_spec_con"] = np.log(ff)
    return dt


def get_sedmspec(z=0., foldername = '20201126_SEDM', 
                 filename = "spec_auto_robot_lstep1__crr_b_ifu20201126_02_43_49_ZTF18abjpmzf.txt",
                 vkernel = 800, t0jd = 58583.2, ebv = 0.022):
    myfile = "../data/spectra/%s/%s"%(foldername, filename)
    tb = asci.read(myfile)
    
    dt = {}
    dt['wave_rest'] = tb['col1'].data/(1+z)
    dt['spec_obs'] = tb['col2'].data
    Aextmag =  extinction.ccm89(dt['wave_rest'], 3.1*ebv, 3.1) # extinction in magnitudes
    tau =  Aextmag / 1.086
    dt['spec_obs0'] = dt['spec_obs'] * np.exp(tau)
    dt["ln_spec_obs"] = np.log(dt['spec_obs0'])

    ww, ff = convolve_with_constant_velocity_kernel(dt['wave_rest'], dt['spec_obs0'], vkernel)
    dt['wave_con'] = ww
    dt['spec_con'] = ff
    dt["ln_spec_con"] = np.log(ff)
    return dt


def psudo_cont_norm(wave, flux, scale=True, spline = False):
    # sp5 = pd.read_csv("../../data/otherSN/SN2011fe/phase13")
    # wave = sp5["wavelength"].values
    # flux = sp5["flux"].values
    
    if spline == True:
        # fit a psudo continuum
        w0 = min(wave)+1000
        w1 = max(wave)-1000
        n = int((w1-w0)//500) + 1
        n = max(n, 2)
        knots = np.linspace(w0, w1, n)
        t,c,k = splrep(wave, flux, k=3, task=-1, t = knots)
        flux_cont = splev(wave, (t,c,k)) 
        ix0 = wave<w0
        flux_cont[ix0] = flux_cont[~ix0][0]
        ix1 = wave>w1
        flux_cont[ix1] = flux_cont[~ix1][-1]
        flux_norm = flux / flux_cont
        if scale==True:
            flux_norm = flux_norm-1
            flux_norm = flux_norm / (np.percentile(abs(flux_norm), 90)*3)
            flux_norm = flux_norm+1
    else:
        # normalize between 5300 -- 5600
        ix = (wave > 5300)&(wave < 5600)
        norm_factor = np.median(flux[ix])
        flux_norm = flux / norm_factor 
    """
    plt.plot(wave, flux)
    plt.plot(wave, flux_cont)
    
    plt.plot(wave, flux_norm)
    """
    return flux_norm
    
    


