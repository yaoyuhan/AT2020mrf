#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:57:42 2021

@author: yuhanyao
"""
import numpy as np
import astropy.io.ascii as asci
from astropy.table import Table, vstack
import astropy.constants as const

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

import matplotlib
import matplotlib.pyplot as plt
fs = 10
matplotlib.rcParams['font.size']=fs


def read_xrt_lc(filename = "GRBs/130427A/lc.qdp"):
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    while lines[0][:4]!="READ":
        lines = lines[1:]
    lines = lines[2:]
    if lines[0][:5] == "!Time":
        lines = lines[1:]
    nl = len(lines)
    
    ts = np.zeros(nl)
    tsp = np.zeros(nl)
    tsm = np.zeros(nl)
    fs = np.zeros(nl)
    fsp = np.zeros(nl)
    fsm = np.zeros(nl)
    
    inds = np.ones(nl, dtype = bool)
    for i in range(nl):
        mylines = lines[i].split("\t")
        if mylines[0][:2] == "NO":
            inds[i] = False
            continue
        if mylines[0][:2] == "! ":
            inds[i] = False
            continue
        if mylines[0][:5] == "!Time":
            inds[i] = False
            continue
        if len(mylines)==1:
            mylines = lines[i].split(" ")
        ts[i] = float(mylines[0])
        tsp[i] = float(mylines[1])
        tsm[i] = -float(mylines[2])
        fs[i] = float(mylines[3])
        fsp[i] = float(mylines[4])
        fsm[i] = -float(mylines[5].replace("\n", ""))
    
    day = 24*3600
    df = Table(data = [ts/day, tsp/day, tsm/day, fs, fsp, fsm],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    
    df = df[inds]
    return df


def add_xlc_sn1998bw(ax):
    data = np.loadtxt("SNe/SN1998bw/data_fig4")
    tt = 10**data[:,0]
    ll = 10**data[:,1]
    t98 = tt[:-1]
    l98 = ll[:-1] - ll[-1]
    color = "gray"
    ax.plot(t98, l98, color = color, zorder = 1, marker = ".", markersize = 1,
            linewidth = 1.5, linestyle = "--", label = "GRB-SNe")
    ax.text(2.2, 3e+40, "980425/98bw", fontsize = fs-1, color = color, rotation = -5)
    
    
def get_xlc_sn2010dh():
    df = read_xrt_lc(filename = "GRBs/100316D/lc.qdp")
    df = df[:-1]
    # Section 2.1 of Margutti+2013
    ts = np.array([13.8, 38.3])
    fs = np.array([5.0e-14, 2.7e-14])
    efs = np.array([1.3e-14, 0.7e-14])
    df1 = Table(data = [ts, np.zeros(2), np.zeros(2), fs, efs, efs],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    return vstack([df, df1])


def get_xlc_03dh():
    # Tiengo+2004, 0.5--2 keV
    ts = np.array([0.2065, 0.2125, 0.21185, 0.2515, 1.2762, 
                   37.3, 60.9, 258.3])
    fs = np.array([153, 146, 162, 134, 9.3, 
                   0.0143, 0.0079, 0.00062]) * 1e-12
    efs = np.array([15, 15, 15, 18, 3,
                    0.0007, 0.0005, 0.00023]) * 1e-12
    # convert from 0.5--2 keV to 0.3--10 keV
    # https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl
    # gamma = 2.17, nh=2e+20
    multi = 2.295
    df = Table(data = [ts, np.zeros(len(ts)), np.zeros(len(ts)), 
                       fs*multi, efs*multi, efs*multi],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    return df


def add_xlc_sn2003dh(ax):
    df = get_xlc_03dh()
    z = 0.1685
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    color = "gray"
    ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right],
                color = color, zorder = 2, fmt = ".-", markersize = 1,
                elinewidth = 0.5, linewidth = 1.5, linestyle = "--")
    ax.text(2.2, 1.8e+44, "030329/03dh", fontsize = fs-1, color = color, rotation = -33)


def get_xlc_sn2006aj():
    df = read_xrt_lc(filename = "SNe/SN2006aj/lc.qdp")
    df = df[df["f+"]!=0]
    # Campana+2006:
    # at ~11.6 day, 0.3--10 keV flux ~ 1.2e-13 erg/cm^2/s
    multi0 = 13
    df["f"] *= multi0
    df["f+"] *= multi0
    df["f-"] *= multi0
    # Soderberg+2006, Chandra
    # but Soderberg assumed the XRT average count rate --> flux converter
    # i.e., the average X-ray spectrum
    # since the spectrum hardened with time
    # the converter should increase
    multi1 = 3
    ts = np.array([8.8, 17.4])
    fs = np.array([4.5e-14, 2.8e-14])*multi1 
    efs = np.array([1.4e-14, 0.9e-14])*multi1 
    df1 = Table(data = [ts, np.zeros(2), np.zeros(2), fs, efs, efs],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    df = vstack([df, df1])
    df = df[np.argsort(df["t"])]
    return df


def add_xlc_sn2006aj(ax):
    df = get_xlc_sn2006aj()
    z = 0.0335
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    color = "gray"
    ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right],
                color = color, zorder = 1, fmt = ".-", markersize = 1,
                elinewidth = 0.5, linewidth = 1.5, linestyle = "--")
    ax.text(2.2, 1e+42, "060218/06aj", fontsize = fs-1, color = color, rotation = -30)
    
    
def add_xlc_sn2010dh(ax):
    df = get_xlc_sn2010dh()
    z = 0.0593
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    color = "gray"
    ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right],
                color = color, zorder = 1, fmt = ".-", markersize = 1,
                elinewidth = 0.5, linewidth = 1.5, linestyle = "--")
    ax.text(2.2, 2.5e+41, "100316D/10dh", fontsize = fs-1, color = color)
    
    
def wmean(d, ivar):
    '''
    >> mean = wmean(d, ivar)
    inverse-variance weighted mean
    '''
    d = np.array(d)
    ivar = np.array(ivar)
    return np.sum(d * ivar) / np.sum(ivar)


def binthem_wmean(x, y, yerr=None, bins=10):
    x_bins = bins.copy()

    xmid = np.zeros(len(x_bins)-1)
    ymid = np.zeros((5,len(x_bins)-1))

    for i in np.arange(len(xmid)):

        ibin = np.where((x>=x_bins[i]) & (x<x_bins[i+1]))[0]
        if i == (len(xmid)-1): # close last bin
            ibin = np.where((x>=x_bins[i]) & (x<=x_bins[i+1]))[0]

        if len(ibin)>0:

            xmid[i] = np.mean(x[ibin])
            ymid[4,i] = np.std(x[ibin])
    
            y_arr = y[ibin]
            ymid[3,i] = len(ibin)
            
            y_arr_err = yerr[ibin]
            ymid[0,i] = wmean(y_arr, 1/y_arr_err*2)

            ymid[[1,2],i] = 1/np.sqrt(sum(1/y_arr_err**2))
            
        else:
            xmid[i] = (x_bins[i] +x_bins[i+1])/2.

        
    if sum(ymid[3,:]) > len(x):
        print ('binthem: WARNING: more points in bins ({0}) compared to lenght of input ({1}), please check your bins'.format(sum(ymid[3,:]), len(x)))
        #key = input()

    return xmid, ymid


def append_130427A_xdeep(df0):
    pdir = '/Users/yuhanyao/Dropbox/Mac/Documents/GitHub/CowAnalogs/paper_Y21/xray/'
    
    dt1 = np.loadtxt(pdir + "GRBs/130427A/DePasquale17_fig1").T
    tt = 10**dt1[0]
    ff = 10**dt1[1]*1e-12
    eff = ff*0.2
    df1 = Table(data = [tt, np.zeros(len(tt)),  np.zeros(len(tt)),
                        ff, eff, eff],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    df = vstack([df0, df1])
    df = df[np.argsort(df["t"])]
    return df


def append_060729_xdeep(df0):
    pdir = '/Users/yuhanyao/Dropbox/Mac/Documents/GitHub/CowAnalogs/paper_Y21/xray/'
    
    dt1 = np.loadtxt(pdir + "GRBs/060729/Grupe10_fig1").T
    day = 3600*24
    tt = 10**dt1[0] / day
    ff = 10**dt1[1]
    eff = ff*0.2
    eff[-2:] = ff[-2:]*0.4
    df1 = Table(data = [tt, np.zeros(len(tt)),  np.zeros(len(tt)),
                        ff, eff, eff],
               names = ["t", "t+", "t-", "f", "f+", "f-"])
    df = vstack([df0, df1])
    df = df[np.argsort(df["t"])]
    return df


def add_grb_lcs(ax, dobin = True, color = "lightskyblue"):
    tb = asci.read("./GRBs/lGRB_sample.dat")
    for i in range(len(tb)):
        grb = "GRB"+tb["GRB"].data[i]
        z = tb["z"].data[i]
        myfile = "./GRBs/sample/%s_lc/flux.qdp"%grb
        df = read_xrt_lc(filename = myfile)
        df = df[df["f-"]!=0]
        if grb == 'GRB130427A':
            df = append_130427A_xdeep(df)
            #print (grb, "append deep xlc", "tmax=", max(df["t"]))
        if grb == "GRB060729":
            df = append_060729_xdeep(df)
        D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
        D_cm = D * const.pc.cgs.value
        multi = 4 * np.pi * D_cm**2
        df["x"] = df["t"] / (1+z)
        df = df[df["x"]>2]
        xx = df["x"].data 
        yy = df["f"].data*multi
        yerr_right = df["f+"].data*multi
        yerr_left = df["f-"].data*multi
    
        if z<0.1:
            print ("  %s ($z = %.5f$),"%(grb, z))
        elif z<1:
            print ("  %s ($z = %.4f$),"%(grb, z))
        else:
            print ("  %s ($z = %.3f$),"%(grb, z))
            
        if dobin == False:
            if i==0:
                ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right],
                            color = color, zorder = 1, fmt = "x-", markersize = 1,
                            elinewidth = 0.3, linewidth = 0.6, linestyle = "-",
                            label = "GRBs")
            else:
                ax.errorbar(xx, yy, yerr = [yerr_left, yerr_right],
                            color = color, zorder = 1, fmt = "x-", markersize = 1,
                            elinewidth = 0.3, linewidth = 0.6, linestyle = "-")
        else:
            # crude way to make bins wider at late-time
            bins_pre = np.arange( 1, 5, 0.1)
            bins_mid1 = np.arange( 5, 40, 0.5)
            bins_mid2 = np.arange( 40, 100, 4)
            bins_late = np.arange( 100, 1e+4, 10)
            bins = np.hstack([bins_pre, bins_mid1, bins_mid2, bins_late])
            
            xbin, ybin = binthem_wmean(xx, yy, 
                                       np.min(np.vstack([yerr_right, yerr_left]), axis = 0), 
                                       bins=bins)
            inz = ybin[3,:]>0
            x, y, ey = xbin[inz], ybin[0,inz], ybin[1,inz]
            
            if i==0:
                ax.errorbar(x, y, ey,
                            color = color, zorder = 1, fmt = "x-", markersize = 1,
                            elinewidth = 0.3, linewidth = 0.6, linestyle = "-",
                            label = "GRBs")
            else:
                ax.errorbar(x, y, ey,
                            color = color, zorder = 1, fmt = "x-", markersize = 1,
                            elinewidth = 0.3, linewidth = 0.6, linestyle = "-")
    print ("%d GRBs plotted"%(i+1))


def get_xlc_sn2010jl():
    """
    Table 5 of Chandra + 2015
    """
    filename = "SNe/SN2010jl/Chandra2015_tab5.dat"
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    tt = []
    ff = []
    eff = []
    
    for i in range(len(lines)):
        myline = lines[i]
        idt = 4
        if myline.split(" ")[4][0] in ["S", "N"]:
            idt = 5
        if myline.split(" ")[5][0] in ["X"]:
            idt = 6
        #print (myline.split(" ")[idt])
        tt.append(float(myline.split(" ")[idt]))
        newsubs = myline.split(" ")[idt+1:]
        if newsubs[0] == '±':
            idf = 2
        else:
            idf = 0
        ff.append(float(newsubs[idf][1:])*1e-13)
        eff.append(float(newsubs[idf+2][:-1])*1e-13)
        
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    #t = np.array([43.55. 53.03, 60.34])
    return df


def get_xlc_sn2005kd():
    """
    Dwarkadas+2016, Table 1
    
    conversion from 0.3--8 keV to 0.3--10 keV
    nh = 4e+21, APEC model, solar metalicity, 17 keV
    """
    multi = 1.18
    tt = np.array([440, 479, 504, 784, 
                   1015, 2200, 2419, 2940])
    ff = np.array([26, 49.6, 41.4, 44.6, 
                   19.86, 6.7, 3.35, 1.98]) * 1e-14 * multi
    eff_right = np.array([13, 27, 4.1, 25,
                         1.87, 0, 0.18, 0.44]) * 1e-14 * multi
    eff_left = np.array([11, 16.8, 9.4, 25.3, 
                        6.93, 0, 1.65, 0.36]) * 1e-14 * multi
    
    df = Table(data = [tt, ff, eff_right, eff_left],
               names = ["t", "f", "f+", "f-"])
    return df


def get_xlc_sn2006jd():
    """
    [1] Chandra+2012, Table 7, unabsorbed 0.2--10 keV flux, 
            multiply by 0.87 to get absorbed 0.3--10 keV flux
    [2] Katsuda+2016, Table 6, unabsorbed 0.2--10 keV flux,
            multiply by 0.87 to get absorbed 0.3--10 keV flux
    """
    multi = 0.87
    #z = 0.0186
    #D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    #D_cm = D * const.pc.cgs.value
    #scale2 = 4 * np.pi * D_cm**2
    tt = np.array([403.2, 431.5, 459.8, 496.2, 698.5, 907.7, 1063.6, 1067.5, 1609.8,
                   #564 / 1.015, 2940 / 1.015
                   ])
    ff = np.array([4.51, 4.28, 4.91, 3.98, 5.35, 3.63, 3.01, 3.38, 3.71,
                   #20.5e+40/scale2*1e+13, 5.7e+40/scale2*1e+13
                   ]) * 1e-13 * multi
    eff_right = np.array([0.73, 0.73, 0.79, 0.66, 0.73, 0.33, 0.60, 0.89, 0.60,
                          #1.5e+40/scale2*1e+13, 0.7e+40/scale2*1e+13
                          ]) * 1e-13 * multi
    eff_left = np.array([0.73, 0.73, 0.79, 0.66, 0.73, 0.30, 0.60, 0.93, 0.60,
                         #1.5e+40/scale2*1e+13, 0.6e+40/scale2*1e+13
                         ]) * 1e-13 * multi
    
    df = Table(data = [tt, ff, eff_right, eff_left],
               names = ["t", "f", "f+", "f-"])
    df = df[np.argsort(df["t"])]
    return df


def get_xlc_scp06f6():
    """
    [1] Leven+2013, XMM, unabsorbed 0.2--10 keV flux, nH = 8.85e+19,
            multiply by 0.71 to get absorbed 0.3--10 keV flux
        CXO non-detection
    """
    #z = 1.189
    # xmm
    #(Time("2006-08-02").mjd - 53767)/(1+1.189)
    # cxo
    #(Time("2006-11-04").mjd - 53767)/(1+1.189)
    tt = np.array([83.1, 126.1])
    ff = np.array([1.3e-13*0.71, 1.4e-14])
    eff = np.array([0.18 * (1.3e-13*0.71), np.nan])
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    df = df[np.argsort(df["t"])]
    return df


def get_xlc_15bn():
    # Margutti+2018 Section 2.1.4
    #z = 0.1136
    # (Time("2015-06-01").mjd - 57013)/(1+z)
    # (Time("2015-12-18").mjd - 57013)/(1+z)
    tt = np.array([144.6, 324.2])
    ff = np.array([9.8e-15, 5.3e-15])
    eff = np.array([np.nan, np.nan])
    df = Table(data = [tt, ff, eff],
               names = ["t", "f", "ferr"])
    df = df[np.argsort(df["t"])]
    return df


def add_SLSNe_xlc(ax):
    color = "plum"

    df = get_xlc_scp06f6()
    z = 1.189
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    t1 = df["t"]
    L1 = df["f"]*multi
    eL1 = df["ferr"]*multi
    ax.errorbar(t1[0], L1[0], eL1[0], fmt = "s-", markersize = 4, 
                elinewidth = 0.6, linewidth = 1,
                color = color, zorder = 4, label = "SLSNe")
    ax.plot(t1[1], L1[1], markersize = 4,
            color = color, zorder = 4, marker = "v", alpha = 0.6)
    ax.plot(t1, L1, linestyle = "-.", color = color, markersize = 0.1, alpha = 0.6)
    ax.text(90, 7e+43, "SCP 06F6", fontsize = fs-1, color = color)
    
    # PTF 12dam
    z = 0.107
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    #(Time("2012-06-11").mjd - 56022)/(1+z) --> 59.0
    #(Time("2012-06-19").mjd - 56022)/(1+z) --> 66.1
    Lx = 7e-16 * multi
    ax.errorbar(62.5, Lx, xerr = 3, yerr = 1e+40, fmt = "s-", markersize = 4, 
                elinewidth = 0.6, linewidth = 1,
                color = color, zorder = 4)
    ax.text(45, 7e+39, "PTF12dam", fontsize = fs-1, color = color)
    
    # SN2015bn
    df = get_xlc_15bn()
    z = 0.1136
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    t1 = df["t"]
    L1 = df["f"]*multi
    ax.plot(t1, L1, markersize = 4, color = color, zorder = 4, marker = "v", 
            linestyle = "-.", alpha = 0.6)
    ax.text(300, 1e+41, "15bn", fontsize = fs-1, color = color)
    

def add_SNeIIn_xlc(ax):
    color = "mediumaquamarine"
    
    df = get_xlc_sn2010jl()
    z = 0.0107
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr = df["ferr"]*multi
    ax.errorbar(xx, yy, yerr, color = color, zorder = 1, fmt = ">:", 
                markersize = 2, linestyle = "-.",
                elinewidth = 0.5, linewidth = 0.6, label = "SNe IIn")
    ax.text(1050, 1.5e+40, "10jl", fontsize = fs-1, color = color)
    
    df = get_xlc_sn2005kd()
    df = df[df["t"]!=479]
    z = 0.015040
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    ax.errorbar(xx, yy, [yerr_left, yerr_right],
                color = color, zorder = 1, fmt = ">:", markersize = 2,
                linestyle = "-.",elinewidth = 0.5, linewidth = 0.6)
    ax.text(1070, 1e+41, "05kd", fontsize = fs-1, color = color)
    
    df = get_xlc_sn2006jd()
    z = 0.0186
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    D_cm = D * const.pc.cgs.value
    multi = 4 * np.pi * D_cm**2
    xx = df["t"]
    yy = df["f"]*multi
    yerr_right = df["f+"]*multi
    yerr_left = df["f-"]*multi
    ax.errorbar(xx, yy, [yerr_left, yerr_right],
                color = color, zorder = 1, fmt = ">:", markersize = 2,
                linestyle = "-.",elinewidth = 0.5, linewidth = 0.6)
    ax.text(1000, 3e+41, "06jd", fontsize = fs-1, color = color)