#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:22:22 2021

@author: yuhanyao
"""
import numpy as np
from astropy.time import Time
import astropy.io.ascii as asci
from astropy.table import Table
import astropy.constants as const

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

import matplotlib
import matplotlib.pyplot as plt
fs = 10
matplotlib.rcParams['font.size']=fs


def cow_xrt_lc():
    # T0 for this burst is Swift MET=551097181.6 s, = 2018 Jun 19 at 10:32:40.470 UT
    t0 = Time("2018-06-19T10:32:40.470").mjd
    tb = asci.read("../data/data_cow/XRT/curve_nosys.qdp")
    names = ["Time", "T_+ve", "T_-ve", "Rate", "Ratepos", "Rateneg", "ObsID"]
    for i in range(len(names)):
        name = names[i]
        tb.rename_column("col%d"%(i+1), name)
    tb["mjd"] = t0 + tb["Time"]/3600/24
    tb["f_XRT"] =  tb["Rate"]*4.26e-11 # This is from Section 2.2.1 of Ho+2019
    tb["f_XRT_unc_right"] =  tb["Ratepos"]*4.26e-11
    tb["f_XRT_unc_left"] =  -tb["Rateneg"]*4.26e-11
    d_cm = 60 * 1e+6 * const.pc.cgs.value # 60 Mpc
    colnames = tb.colnames[-3:]
    for i in range(len(colnames)):
        colname_old = colnames[i]
        colname_new = "L"+colname_old[1:]
        tb[colname_new] = tb[colname_old] * 4 * np.pi * d_cm**2 
    
    tb["phase"] = tb["mjd"]-58285.44
    return tb


def load_sn2006jc_xlc():
    """
    Immler+2008, Table 1
    unabsorbed 0.2--10 keV
    But since nh = 1.45e+20 cm^-2; the difference is negligiable
    """
    tt = np.array([40, (50+19)/2, (53+94)/2, (126+98)/2, (141+128)/2, (159+143)/2, (183+171)/2])
    ett = np.array([0, (50-19)/2, (94-53)/2, (126-98)/2, (141-128)/2, (159-143)/2, (183-171)/2])
    ff = np.array([0.8e-14, 1.2e-14, 3.3e-14, 5.2e-14, 2.0e-14, 1.7e-14, 1.5e-14])
    eff = np.array([0.3e-14, 0.8e-14, 0.7e-14, 1.0e-14, 0.7e-14, 0.7e-14, 0.6e-14])
    df = Table(data = [tt, ett, ett, ff, eff, eff],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    return df


def load_sn2010al_xlc():
    """
    X-ray data from Ofek + 2013; Figure 5
    Classification of SN Ibn comes from Pastorello+2015
    unabsorbed 0.2--10 keV
    But since nh = 3.92e+20 cm^-2; the difference is negligiable
    """
    tt = np.array([15, 30, 47])
    ett = np.array([5, 9, 7])
    z = 0.0075
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    ff = np.array([1.7e+40, 5.1e+40, 1.9e+40]) / multi
    eff_right = np.array([5e+40, 8e+40, 6e+40]) / multi
    eff_left = np.array([1.2e+40, 3.5e+40, 1.2e+40]) / multi
    df = Table(data = [tt, ett, ett, ff, eff_right, eff_left],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    return df


def add_SNeIbn_xlc(ax):
    color = "darkcyan"
    df = load_sn2006jc_xlc()
    distance_cm = 24*1e+6 * const.pc.cgs.value
    multi = 4*np.pi*distance_cm**2
    xx = df["t"]
    xerr = df["t+"]
    yy = df["f"]*multi
    yerr = df["f+"]*multi
    ax.errorbar(xx, yy, yerr = yerr, xerr = xerr,
                color = color, zorder = 1, fmt = "<:", 
                markersize = 3, elinewidth = 0.3, linewidth = 1, label = "SNe Ibn")
    ax.text(200, 1e+39, "06jc", fontsize = fs-1, color = color)
    
    df = load_sn2010al_xlc()
    z = 0.0075
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    xerr = df["t+"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right], xerr = xerr,
                color = color, zorder = 1, fmt = "<:",
                markersize = 3, elinewidth = 0.3, linewidth = 1)
    ax.text(15, 1e+40, "10al", fontsize = fs-1, color = color)
    
    
def load_1644_lc():
    dt = asci.read("./TDEs/SwiftJ1644+57/Mangano2016_tab2.dat")
    tt_left = dt["col1"].data # Starting time of interval, relative to BAT trigger
    tt_right = dt["col2"].data # End time of interval, relative to BAT trigger
    tt = (tt_left + tt_right) / 2.
    ett_left = tt - tt_left
    ett_right = tt_right - tt
    ff = dt["col3"].data
    eff = dt["col4"].data
    df = Table(data = [tt, ett_right, ett_left, ff, eff, eff],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    return df

    
def add_tde_lcs(ax, toBrad = False):
    color = "b"
    
    df = load_1644_lc()
    z = 0.35
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    
    yy = df["f"] * multi
    yerr_left = df["f-"] * multi
    yerr_right = df["f+"] * multi
    
    xx = df["t"] / (1+z) # correct to rest-frame
    xerr_left = df["t-"] / (1+z) 
    xerr_right = df["t+"]  / (1+z)
    
    ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right], 
                xerr = [xerr_left, xerr_right],
                color = color, zorder = 1, marker = "+", markersize = 1,
                elinewidth = 0.3, linewidth = 0.3, linestyle = ":")
    if toBrad:
        ax.text(60, 2e+46, "SwiftJ1644", color = color, fontsize = fs, weight="bold")
    else:
        ax.text(60, 2e+46, "SwiftJ1644", color = color, fontsize = fs-1)
    