#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:20:28 2021

@author: yuhanyao
"""
import os
import time
import numpy as np
import pandas as pd
import extinction
import subprocess
from copy import deepcopy
import astropy.constants as const
from astropy.io import fits
import astropy.io.ascii as asci
from astropy.time import Time
from astropy.table import Table, vstack
from collections import OrderedDict as odict

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

#from helper.app2abs import add_datecol, add_physcol
from load_opt_phot import load_ztf_lc 


import matplotlib
import matplotlib.pyplot as plt
fs= 10
matplotlib.rcParams['font.size']=fs
ms = 6
matplotlib.rcParams['lines.markersize']=ms


def get_date_span(tb):
    mjdstart = tb['mjd'].values[0]
    mjdend = tb['mjd'].values[-1]
    mjds = np.arange(mjdstart, mjdend+0.9)
    t = Time(mjds, format = "mjd")
    datetime64 = np.array(t.datetime64, dtype=str)
    space = " "
    dates = [space.join(x.split('T')[0].split('-')) for x in datetime64]
    dates = np.array(dates)
    return dates


def add_datecol(tb):
    """
    tb is pandas dataframe
    
    columns that must exist: mjd
    """
    t = Time(tb["mjd"].values, format='mjd')
    tb['datetime64'] = np.array(t.datetime64, dtype=str)
    space = " "
    date = [space.join(x.split('T')[0].split('-')) for x in tb['datetime64'].values]
    tb['date'] = date
    
    tb = tb.sort_values(by = "mjd")
    return tb


def add_physcol(tb, magcol = 'mag0_abs'):
    """
    tb is pandas dataframe35*
    
    columns that must exist: 
        wave: in angstrom
        magcol (e.e.g: mag0_abs): extinction corrected absolute magnitude
        emag: uncertainty in magnitude
        
    please avoid mag == 99 or any invalid values...
    """
    # zero point in AB magnitude: 3631 Jy 
    # 1 Jy = 1e-23 erg / s / Hz / cm^{-2}
    if "wave" in tb.columns:
        tb['freq'] = const.c.cgs.value / (tb['wave'].values * 1e-8) # Hz
    elif "freq" in tb.columns:
        tb['wave'] = const.c.cgs.value / tb['freq'].values * 1e8 # Hz

    tb['fratio'] = 10**(-0.4 * tb[magcol].values)
    tb['fratio_unc'] = np.log(10) / 2.5 * tb['emag'].values * tb['fratio'].values
    
    fnu0 = 3631e-23 # erg / s/ Hz / cm^2
    tb['fnu'] = tb['fratio'].values * fnu0 # erg / s/ Hz / cm^2
    tb['fnu_unc'] = tb['fratio_unc'].values * fnu0 # erg / s/ Hz / cm^2
    
    tb['nufnu'] = tb['fnu'].values * tb['freq'].values # erg / s / cm^2
    tb['nufnu_unc'] = tb['fnu_unc'].values * tb['freq'].values # erg / s / cm^2
    
    tb['flambda'] = tb['nufnu'].values / tb['wave'] # erg / s / cm^2 / A
    tb['flambda_unc'] = tb['nufnu_unc'].values / tb['wave'] # erg / s / cm^2 / A
    
    tb['Llambda'] = tb['flambda'].values * 4 * np.pi * (10*const.pc.cgs.value)**2 # erg / s / A
    tb['Llambda_unc'] = tb['flambda_unc'].values * 4 * np.pi * (10*const.pc.cgs.value)**2 # erg / s / A
    return tb



def get_at2019dge(colorplt=True):
    t_max = 58583.2 
    t_fl = t_max - 2.91
    z = 0.0213
    ebv = 0.022
    tspecs = np.array([58583.59659, # Keck spec JD at midpoint
                       58597.46300334, # DBSP spec
                       58582.146159,
                       58583.129278,
                       58595.213889,
                       58668.492577])
    
    # LT, SEDM, P48
    tb = pd.read_csv('./color_data/lc_at2019dge.csv')
    result = odict([('z', z),
                    ('ebv', ebv),
                    ('t_max', t_max),
                    ('tspecs', tspecs),
                    ("tb", tb)])
    tb = tb[tb.instrument!="P60+SEDM"]
    
    if colorplt==False:
        return result
    else:
        ix = np.any([tb["instrument"].values == "P48",
                     tb["instrument"].values == "LT+IOO"], axis=0)
        tb = tb[ix]
        ix = np.in1d(tb["filter"].values, np.array(['g', 'r', 'i', 'z']))
        tb = tb[ix]
        
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            rmag = 99
            gmag = 99
            imag = 99
            zmag = 99
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            ztb = tbsub[tbsub["filter"].values=="z"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(ztb)!=0:
                zmjds = ztb["mjd"].values
                zmags = ztb["mag0"].values
                zemags = ztb["emag"].values
                zwtgs = 1/zemags**2
                zmag = np.sum(zmags * zwtgs) / np.sum(zwtgs)
                zmjd = np.sum(zmjds * zwtgs) / np.sum(zwtgs)
                zemag = 1/ np.sqrt(np.sum(zwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            if len(itb)!=0 and len(ztb)!=0:
                mcolor.append(imag - zmag)
                mjds.append( 0.5 * (imjd + zmjd) )
                mcolor_unc.append( np.sqrt(iemag**2 + zemag**2) )
                colorname.append("imz")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb['phase_rest'] = (ctb['mjd'] - t_fl) / (1+z)
        ctb = ctb.to_pandas()
        
        result.update({"ctb": ctb})
        return result
    
    
def get_iPTF14gqr(colorplt=False):
    """
    De+18, Table S1, already corrected for extinction
    """
    z = 0.063
    # ebv = 0.082
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_exp = 56943.74 # 
    t_max = 56950.26 # g band max light + 3
    
    tb = Table(fits.open('./color_data/lc_iptf14gqr.fit')[1].data)
    tb.rename_column('MJD' , 'mjd')
    #tb['texp_rf'] = (tb['mjd'] - t_exp) / (1+z)
    tb['phase_rest'] = (tb['mjd'] - t_exp) / (1+z)
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    # tb = tb[tb["Filt"]=="g   "]
    tb = tb[~np.isnan(tb['e_mag'])]
    tb.rename_column('Filt' , 'filter')
    tb.rename_column('e_mag' , 'emag')
    tb.rename_column('mag' , 'mag0')
    
    ixg = tb['filter']=="g   "
    ixB = tb['filter']=="B   "
    ixV = tb['filter']=="V   "
    ixr = tb['filter']=="r   "
    ixi = tb['filter']=="i   "
    ixUVW1 = tb['filter']=="UVW1"
    ixUVW2 = tb['filter']=="UVW2"
    
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixUVW2] = 2079
    tb['wave'][ixUVW1] = 2614
    tb['wave'][ixB] = 4359
    tb['wave'][ixg] = 4814
    tb['wave'][ixV] = 5430
    tb['wave'][ixr] = 6422
    tb['wave'][ixi] = 7883
    
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    
    tb = tb.to_pandas()
    #tb["texp_rf"] = tb["Phase"]
    tb = tb.drop(columns=["recno", "Phase", "l_mag"])
    """
    ix = np.any([tb['Tel'].values=="P60 ",
                 tb["filter"].values=='g   '], axis=0)
    tb = tb[ix]
    """
    tb = add_datecol(tb)
    tb = add_physcol(tb)
    tt = tb["tmax_rf"].values
    epochs = ["        " for x in range(len(tt))]
    epochs = np.array(epochs)
    """
    ix = (tt>-5.6)&(tt<-5.55)
    epochs[ix] = "epoch 01"
    """
    ix = (tt>-5.55)&(tt<-5.50)
    epochs[ix] = "epoch 02"
    
    ix = (tt>-5.50)&(tt<-5.45)
    epochs[ix] = "epoch 03"
    
    ix = (tt>-5.2)&(tt<-5.0)
    epochs[ix] = "epoch 04"
    ix = (tt>-5.0)&(tt<-4.7)
    epochs[ix] = "epoch 05"
    
    ix = (tt>-4.7)&(tt<-4.5)
    epochs[ix] = "epoch 06"
    ix = (tt>-4.5)&(tt<-3.5)
    epochs[ix] = "epoch 07"
    ix = (tt>-3.5)&(tt<-2.5)
    epochs[ix] = "epoch 08"
    ix = (tt>-1.5)&(tt<-1)
    epochs[ix] = "epoch 09"
    ix = (tt>-1)&(tt<-0.82)
    epochs[ix] = "epoch 10"
    ix = (tt>-0.82)&(tt<-0.6)
    epochs[ix] = "epoch 11"
    ix = (tt>-0.5)&(tt<0.5)
    epochs[ix] = "epoch 12"
    ix = (tt>0.5)&(tt<1.5)
    epochs[ix] = "epoch 13"
    ix = (tt>1.5)&(tt<2.5)
    epochs[ix] = "epoch 14"
    ix = (tt>3.5)&(tt<4.5)
    epochs[ix] = "epoch 15"
    ix = (tt>4.5)&(tt<5)
    epochs[ix] = "epoch 16"
    ix = (tt>5)&(tt<5.6)
    epochs[ix] = "epoch 17"
    ix = (tt>5.6)&(tt<5.8)
    epochs[ix] = "epoch 18"
    ix = (tt>6)&(tt<7)
    epochs[ix] = "epoch 19"
    ix = (tt>7)&(tt<8)
    epochs[ix] = "epoch 20"
    ix = (tt>8)&(tt<9)
    epochs[ix] = "epoch 21"
    tb["epoch"] = epochs

    if colorplt==False:
        return tb
    else:
        tb = add_datecol(tb)
        ix = np.in1d(tb["filter"].values, np.array(['g   ', 'r   ', 'i   ']))
        tb = tb[ix]

        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r   " in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g   "]
            rtb = tbsub[tbsub["filter"].values=="r   "]
            itb = tbsub[tbsub["filter"].values=="i   "]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        ctb['phase_rest'] = (ctb['mjd'] - t_exp) / (1+z)
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb
    
    
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
    #print ("MJD max = %d, min = %d"%(mjd2, mjd1))

    xbound = np.arange(mjd1, mjd2+1)
    nbin = len(xbound)-1
    xs = np.ones(nbin)
    ys = np.ones(nbin)
    eys  = np.ones(nbin)
    for j in range(nbin):
        xmin = xbound[j]
        xmax = xbound[j+1]
        ix = (mjd>=xmin)&(mjd<xmax)
        #print ("MJD max = %d, min = %d, %d data"%(xmax, xmin, np.sum(ix)))
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
    
    
def load_ipac_lc(name, basecorr = False, baseline_end = 58800,
                 doplot = False, dobin = True, load_other = True):
    myfile = "./color_data/ipac_lcs/%s_ipac.txt"%name
    f = open(myfile)
    lines = f.readlines()
    f.close()
    tb = asci.read(lines[57:])
    colnames = (lines[55][1:].split('\n'))[0].split(', ')
    for j in range(len(colnames)):
        tb.rename_column('col%d'%(j+1), colnames[j])   
    if tb['forcediffimfluxunc'].dtype in ['<U16', '<U17', '<U18', '<U19']:
        ix = tb['forcediffimfluxunc']=='null'
        print (np.sum(ix), "nulls")
        tb = tb[~ix]
        tb['forcediffimfluxunc'] = np.array(tb['forcediffimfluxunc'], dtype=float)
    tb['forcediffimflux'] = np.array(tb['forcediffimflux'], dtype=float)
    tb = tb.to_pandas()
    
    tb.rename(columns={'forcediffimflux':'Fpsf',
                       'forcediffimfluxunc':'Fpsf_unc',
                       'zpdiff':'zp',
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
    if name == "ZTF20abfhyil":
        print ("Only use public data for %s!"%name)
        tb = tb[tb.programid==1]
    
    ix_remove = tb["seeing"]>4
    frac_remove =  np.sum(ix_remove)/len(ix_remove)
    print ("ZTF Select seeing < 4 --> remove %.2f percent of data"%(frac_remove*100))
    if frac_remove < 0.05:
        tb = tb[~ix_remove]
    else:
        print (" Skipping this step")
        
    ix_remove = tb["scisigpix"]>25
    frac_remove =  np.sum(ix_remove)/len(ix_remove)
    print ("ZTF Select scisigpix < 25 --> remove %.2f percent of data"%(frac_remove*100))
    if frac_remove < 0.05:
        tb = tb[~ix_remove]
    else:
        print (" Skipping this step")
    
    ix_remove = tb["infobitssci"]!=0
    frac_remove =  np.sum(ix_remove)/len(ix_remove)
    print ("ZTF Select infobitssci == 0 --> remove %.2f percent of data"%(frac_remove*100))
    if frac_remove < 0.05:
        tb = tb[~ix_remove]
    else:
        print (" Skipping this step")
        
    tb['fcqfid'] = tb['field']*10000 + tb['ccdid']*100 + tb['qid']*10 + tb['filterid']
    if basecorr == True:
        tb = Table.from_pandas(tb)
        print ("Perform ZTF baseline correction")
        ixbase = tb["mjd"].data < baseline_end
        tb_base = deepcopy(tb[ixbase])
        tb_after = deepcopy(tb[~ixbase])
        fcqfids = np.unique(tb_base['fcqfid'].data)

        ycol = "Fpsf"
        eycol = "Fpsf_unc"
        if doplot:
            plt.figure(figsize = (15, 7))
            ax = plt.subplot(121)
            ax2 = plt.subplot(122)
        for i in range(len(fcqfids)):
            fcqfid = fcqfids[i]
            # manually turn off r-band correction for this source
            # since fcqfid = 14621032, nobs = 46 --> mean = 116.44 +- 6.13 DN, chi2_red = 9.10
            ix = tb_base["fcqfid"] == fcqfid
            nobs = np.sum(ix)
            tb_fcqf = deepcopy(tb_base[ix])
            refmjdstart = tb_fcqf['refjdstart'][0] - 2400000.5
            refmjdend = tb_fcqf['refjdend'][0] - 2400000.5
            
            flux = tb_fcqf[ycol]
            flux_unc = tb_fcqf[eycol]
            weight = 1/flux_unc**2
            wmean = np.sum(flux * weight) / np.sum(weight)
            wmean_unc = np.sqrt(1 / np.sum(weight))
            chi2_red = np.sum((flux - wmean)**2 / flux_unc**2) / (len(flux)-1)
            if ycol == "Fratio":
                print ("fcqfid = %d, nobs = %d --> Fratio mean = %.1f +- %.1f, chi2_red = %.2f"%(fcqfid, nobs, wmean, wmean_unc, chi2_red))
                 
            elif ycol == "Fpsf":
                print ("fcqfid = %d, nobs = %d --> Fpsf mean = %.2f +- %.2f e-9 chi2_red = %.2f"%(fcqfid, nobs, wmean*1e+9, wmean_unc*1e+9, chi2_red))
            print ("             Ref building mjd: %d -- %d"%(refmjdstart, refmjdend))
            
            print ("Performing offset correction...")
            tb[ycol][tb["fcqfid"]==fcqfid] -= wmean
            print ("Performing scaling correction...")
            tb[eycol][tb["fcqfid"]==fcqfid] *= np.sqrt(chi2_red)
                
            if doplot:
                ix1 = tb_fcqf["programid"]==1
                ix2 = tb_fcqf["programid"]==2
                ix3 = tb_fcqf["programid"]==3
                print (np.sum(ix1))
                print (np.sum(ix2))
                print (np.sum(ix3))
                p = ax.errorbar(tb_fcqf["mjd"][ix2], flux[[ix2]], flux_unc[[ix2]], 
                                fmt = "o", label = "fcqfid = %d, pid = 2"%fcqfid,
                                markersize = 3, elinewidth = 0.2)
                color = p[0].get_color()
                if np.sum(ix1)>0:
                    ax.errorbar(tb_fcqf["mjd"][ix1], flux[[ix1]], flux_unc[[ix1]], 
                                fmt = "x", label = "pid = 1", color = color,
                                markersize = 5, elinewidth = 0.2)
                if np.sum(ix3)>0:
                    ax.errorbar(tb_fcqf["mjd"][ix3], flux[[ix3]], flux_unc[[ix3]], 
                                fmt = "*",label = "pid = 3", color = color,
                                markersize = 5, elinewidth = 0.2)
                
                if ycol == "Fratio":
                    ypos = 5e-8 - i*1e-8
                    yoff = 1e-9
                elif ycol == "Fpsf":
                    ypos = 300 + i * 50
                    yoff = 5
                ax.plot([refmjdstart, refmjdend], [ypos, ypos], color = color)
                xlims = [58200, baseline_end]
                ax.plot([xlims[0], xlims[1]], [wmean, wmean], color = color, linestyle = "--")
                if ycol == "Fratio":
                    ax.text(58200, ypos+yoff, "Fratio mean = %.2f +- %.2f e-9 chi2_red = %.2f"%(wmean*1e+9, wmean_unc*1e+9, chi2_red))
                elif ycol == "Fpsf":
                    ax.text(58200, ypos+yoff, "Fpsf mean = %.1f +- %.1f, chi2_red = %.2f"%(wmean, wmean_unc, chi2_red))
        
            ix2 = tb_after["fcqfid"] == fcqfid
            tb2_fcqf = tb_after[ix2]
            flux2 = tb2_fcqf[ycol]
            flux2_unc = tb2_fcqf[eycol]
            
            if doplot:
                ix1 = tb2_fcqf["programid"]==1
                ix2 = tb2_fcqf["programid"]==2
                ix3 = tb2_fcqf["programid"]==3
                print (np.sum(ix1))
                print (np.sum(ix2))
                print (np.sum(ix3))
                
                if np.sum(ix1)>0:
                    ax2.errorbar(tb2_fcqf["mjd"][ix1], flux2[[ix1]], flux2_unc[[ix1]], 
                                fmt = "x", label = "pid = 1", color = color,
                                markersize = 5, elinewidth = 0.2)
                if np.sum(ix3)>0:
                    ax2.errorbar(tb2_fcqf["mjd"][ix3], flux2[[ix3]], flux2_unc[[ix3]], 
                                fmt = "*", label = "pid = 3", color = color,
                                markersize = 5, elinewidth = 0.2)
                if np.sum(ix2)>0:
                    ax2.errorbar(tb2_fcqf["mjd"][ix2], flux2[[ix2]], flux2_unc[[ix2]], 
                                fmt = "o", label = "fcqfid = %d, pid = 2"%fcqfid,
                                markersize = 3, elinewidth = 0.2)
                
        if doplot:  
            ax.legend(ncol = 2, loc = "lower left")
            plt.tight_layout()
            
        if ycol == "Fpsf":
            # baseline correction performed on the Fpsf unit --> re-calcualte Fratio
            F0 = 10**(tb['zp']/2.5)
            eF0 = F0 / 2.5 * np.log(10) * tb['ezp']
            Fpsf = tb['Fpsf']
            eFpsf = tb['Fpsf_unc']
            Fratio = Fpsf / F0
            eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
            eFratio = np.sqrt(eFratio2)
            tb['Fratio'] = Fratio
            tb['Fratio_unc'] = eFratio
        tb = tb.to_pandas()
    
    tb = tb.drop(columns=["infobitssci", "procstatus", "chi2_red", "seeing", "scisigpix",
                         'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp',
                         "jdobs", 'forcediffimfluxap', 'forcediffimfluxuncap', 'forcediffimsnrap',
                         'forcediffimsnr', 'exptime', 'adpctdif1', 'adpctdif2', 'aperturecorr',
                         'clrcoeff', 'clrcoeffunc', "zpmaginpscirms", "rfid", "ncalmatches", 
                         "scibckgnd"])
    tb = tb[tb.Fratio_unc<1e-8]
    
    tb["uJy"] = tb["Fratio"]*3631e+6
    tb["duJy"] = tb["Fratio_unc"]*3631e+6
    
    ixg = tb["filterid"].values==1
    ixr = tb["filterid"].values==2
    tb["F"] = "g"
    tb["F"].values[ixr] = "r"
    
    tb = tb.drop(columns=['index', 'field', 'ccdid', 'qid', 'filter', 'pid', 'zpmaginpsci', 'ezp',
                          'diffmaglim', 'zp', 'programid', 'Fpsf', 'Fpsf_unc', 'dnearestrefsrc',
                          'refjdstart', 'refjdend', 'filterid'])
    if dobin:
        tbg = bin_lc(tb[ixg])
        tbg["F"] = "g"
        tbr = bin_lc(tb[ixr])
        tbr["F"] = "r"
        tb = pd.concat([tbg, tbr])
        tb = tb.sort_values(by="mjd")
        tb["Fratio"] = tb["uJy"].values / (3631e+6)
        tb["Fratio_unc"] = tb["duJy"].values / (3631e+6)    
        
    tb["Telescope"] = "P48+ZTF"
    
    print ("")
    print ("Loading photometry from other facilities...")
    if load_other == True:
        tb2 = load_other_lc(name)
        if len(tb2) != 0:
            print ("%d rows loaded!"%(len(tb2)))
            tb2 = tb2.drop(columns = ["Unnamed: 0", "JD", "Mag", "eMag"])
            tb2 = tb2.rename(columns = {"Flux": "uJy",
                                  "eFlux": "duJy",
                                  "Filter": "F"})
            tb2["Fratio"] = tb2["uJy"].values / (3631e+6)
            tb2["Fratio_unc"] = tb2["duJy"].values / (3631e+6)    
            tb = pd.concat([tb, tb2])
    return tb


def load_other_lc(name, texp = None):
    filename = "./color_data/anna_lc/combined/combined_lc_%s.txt"%name
    tb2 = pd.read_csv(filename)
    tb2 = tb2[tb2.Telescope!="P48"]
    ixg = tb2["Filter"].values=="g"
    ixr = tb2["Filter"].values=="r"
    ix = ixg | ixr
    tb2 = tb2[ix]
    tb2["mjd"] = tb2["JD"] - 2400000.5
    #ix = tb2[]
    if texp is not None:
        ix = (tb2["mjd"] > texp)&(tb2["mjd"] < (texp+100))
        tb2 = tb2[ix]
    return tb2


def look_otherphot():
    tb = asci.read("./color_data/ho2021a_gold.dat")
    for i in range(len(tb)):
        name = tb["ztfname"][i]
        texp = tb["texp"][i]
        tb2 = load_other_lc(name, texp = texp)
        print (name)
        print (tb2)

def calc_color(tb, xbounds = None):
    """
    calculate g-r color for ZTF object
    """
    mjd = tb["mjd"].values
    Fs = tb["F"].values
    if xbounds is None:
        # default is daily binnning
        mjd1 = int(np.floor(min(mjd)))
        mjd2 = int(np.ceil(max(mjd)))
        xbounds = np.arange(mjd1, mjd2+1)
    print ("bounds: ", xbounds)
        
    ycol = "uJy"
    eycol = "duJy"
    uJy = tb[ycol].values
    duJy = tb[eycol].values
    
    nbin = len(xbounds)-1
    xs = np.ones(nbin)
    exs = np.ones(nbin)
    ys = np.ones(nbin)
    eys  = np.ones(nbin)
    for j in range(nbin):
        xmin = xbounds[j]
        xmax = xbounds[j+1]
        ixg = (mjd>=xmin)&(mjd<xmax)&(Fs == "g")
        ixr = (mjd>=xmin)&(mjd<xmax)&(Fs == "r")
        if np.sum(ixg) == 0:
            continue
        if np.sum(ixr) == 0:
            continue
        mjdg_ = mjd[ixg]
        mjdr_ = mjd[ixr]
        fg_uJy_ = uJy[ixg]
        fr_uJy_ = uJy[ixr]
        fg_duJy_ = duJy[ixg]
        fr_duJy_ = duJy[ixr]
        if np.sum(ixg)>1:
            mjdg = np.median(mjdg_)
            weightg = 1 / fg_duJy_**2
            fg_uJy = np.sum(weightg * fg_uJy_) / np.sum(weightg)
            fg_duJy = np.sqrt(1. / np.sum(weightg))
        else:
            mjdg = mjdg_
            fg_uJy = fg_uJy_
            fg_duJy = fg_duJy_
        if np.sum(ixr)>1:
            mjdr = np.median(mjdr_)
            weightr = 1 / fr_duJy_**2
            fr_uJy = np.sum(weightr * fr_uJy_) / np.sum(weightr)
            fr_duJy = np.sqrt(1. / np.sum(weightr))
        else:
            mjdr = mjdr_
            fr_uJy = fr_uJy_
            fr_duJy = fr_duJy_
        if fg_uJy < 0:
            continue
        if fr_uJy < 0:
            continue
        
        xs[j] = (max(max(mjdg_), max(mjdr_)) + min(min(mjdg_), min(mjdr_)))/2.
        exs[j] = (max(max(mjdg_), max(mjdr_)) - min(min(mjdg_), min(mjdr_)))/2.
        # calcualte the color
        gmr = -2.5 * np.log10(fg_uJy / fr_uJy)
        gmr_unc = 2.5 / np.log(10) * (fr_uJy**2 * fg_duJy**2 + fg_uJy**2 * fr_duJy**2)**0.5 / (fg_uJy * fr_uJy)
        ys[j] = gmr
        eys[j] = gmr_unc
        
    ix = xs!=1
    xs = xs[ix]
    exs = exs[ix]
    ys = ys[ix]
    eys = eys[ix]
    tb_new = Table(data = [xs, exs, ys, eys], 
                   names = ["mjd", "mjd_unc", "gmr", "gmr_unc"])
    return tb_new


def load_lc(name, z=None, texp=None):
    xbounds = None
    texp_ = texp
    texp = int(texp)
    if name == "ZTF20abfhyil":
        tb = load_ipac_lc(name, load_other = False)
        ix = (tb["mjd"].values>59012)&(tb["mjd"].values<(59012+50))
        tb = tb[ix]
        z = 0.1353
        texp = 59012
        xbounds = np.arange(texp, texp+20, 1)
        #xbounds = np.hstack([xbounds, np.array([59037, 59043, 59050, 59058])])
        xbounds = np.hstack([xbounds, np.array([59037, 59043, 59058])])
    elif name == "ZTF20aclfmwn":
        tb = load_ipac_lc(name)
        # checked
        tb = tb[tb["Fratio_unc"]<5e-9]
        ix = (tb["mjd"].values>59135)&(tb["mjd"].values<(59135+50))
        tb = tb[ix]
        xbounds = np.array([59140,59141,59142,59143,59144,59145,59146,
                            59156,59160,59185])
    elif name == "ZTF20acigusw":
        tb = load_ipac_lc(name, basecorr = True,
                          baseline_end = 59100, doplot = False)
        tb = tb[tb["Fratio_unc"]<5e-9]
        tb = tb[tb["mjd"]>59100]
        tb = tb[tb["mjd"]<59300]
        #ix = (tb["mjd"].values>59118)&(tb["mjd"].values<(59128+100))
        xbounds = np.array([59134,59135,59136,59137,59138,59140,
                            59141,
                            59144,59146,59148,59153,59163,59175,59193])
    elif name == "ZTF18aakuewf":
        tb = load_ipac_lc(name, basecorr = True,
                          baseline_end = 58220, doplot = False,
                          dobin = False)
        tb = tb[tb["fcqfid"]!=6791423]
        tb = tb[tb["mjd"]<58300]
        tb = tb[tb["mjd"]>58200]
        ixno = (tb["mjd"].values>58254)&(tb["mjd"].values<58255)
        tb = tb[~ixno]
        ixno = (tb["mjd"].values>58257)&(tb["mjd"].values<58258)
        tb = tb[~ixno]
        xbounds = np.array([58229, 58230, 58231, 58232, 58233,
                            58242, 58245, 58247, 58250, 58254, 
                            58264])
    elif name == "ZTF18abcfcoo":
        tb = load_ipac_lc(name, basecorr = True,
                          baseline_end = 58285, doplot= False,
                          dobin = False)
        xbounds = np.arange(texp, texp+30, 1)
        xbounds = np.hstack([xbounds, np.array([58321, 58325, 58326, 58331])])
    elif name == "ZTF18abukavn":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58440]
        xbounds = np.arange(texp, texp+35, 1)
        xbounds = np.hstack([xbounds, np.array([58410, 58422, ])])
    elif name == "ZTF18abvkmgw":
        tb = load_ipac_lc(name, basecorr = True,
                          baseline_end = 58372.5, doplot = False,
                          dobin = False)
        tb = tb[tb["mjd"]<58600]
        xbounds = np.arange(58373, 58373+12, 1)
        xbounds = np.hstack([xbounds, np.array([58386, 58390])])
    elif name == "ZTF18abwkrbl":
        tb = load_ipac_lc(name, basecorr = True,
                          baseline_end = 58376, doplot = False,
                          dobin = False)
        tb = tb[tb["fcqfid"]!=16490521]
        tb = tb[tb["fcqfid"]!=16490522]
        tb = tb[tb["mjd"]<58500]
        xbounds = np.arange(texp, texp+20, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58400,58420,58424,58430, 58440])])
    elif name == "ZTF19aakssbm":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58800]
        ixno = (tb["mjd"] > 58559.2)&(tb["mjd"] < 58560)
        tb = tb[~ixno]
        ixno = (tb["mjd"] > 58562)&(tb["mjd"] < 58565)
        tb = tb[~ixno]
        ixno = (tb["mjd"] > 58568)&(tb["mjd"] < 58570)
        tb = tb[~ixno]
        xbounds = np.arange(texp, texp+10, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58550, 58555,58560,58575,
                                     58584, 58590, 58600])])
    elif name == "ZTF19aapfmki":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58800]
        #xbounds = np.arange(texp, texp+30, 1)
        #xbounds = np.hstack([xbounds, np.array([58618,58634,58637, 58644])])
        xbounds = np.arange(texp, texp+21, 1)
        xbounds = np.hstack([xbounds, np.array([58604, 58608,58618,58644])])
    elif name == "ZTF19abobxik":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58750]
        xbounds = np.arange(texp, texp+20, 2)
        xbounds = np.hstack([xbounds, 
                             np.array([58721, 58723, 58728, 58732])])
    elif name == "ZTF19abuvqgw":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58900]
        tb = tb[tb["mjd"]>58700]
        xbounds = np.arange(texp, texp+10, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58736, 58740, 58747, 58755,  58763])])
    elif name == "ZTF19abyjzvd":
        tb = load_ipac_lc(name)
        
        tb = tb[tb["mjd"]<58900]
        tb = tb[tb["mjd"]>58725]
        xbounds = np.arange(texp+1, texp+21, 2)
        xbounds = np.hstack([xbounds, 
                             np.array([58755, 58758])])
    elif name == "ZTF19acayojs":
        xbounds = np.arange(texp, texp+15, 1)
        tb = load_ipac_lc(name)
        xbounds = np.hstack([xbounds, 
                             np.array([58764, 58765, 58770])])
    elif name == "ZTF19accjfgv":
        tb = load_ipac_lc(name, basecorr = True, doplot = False,
                          dobin = False, baseline_end = 58700)
        tb = tb[tb["mjd"]<58840]
        xbounds = np.arange(texp, texp+16, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58780, 58787])])
    elif name == "ZTF20aaelulu":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<(texp+100)]
    elif name == "ZTF20aahfqpm":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<(texp+60)]
        xbounds = np.arange(texp, texp+4, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58876, 58877, 58879, 58880,
                                       58882, 58885,
                                     58890, 58893, 58897,
                                       58901,58902,58907, 58915])])
    elif name == "ZTF20aaxhzhc":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<59080]
        xbounds = np.arange(texp, texp+59, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([59035, 59045])])
    elif name == "ZTF20aayrobw":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<59100]
    elif name == "ZTF20aazchcq":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<59040]
        xbounds = np.arange(texp, texp+20, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([59000, 59020])])
    elif name == "ZTF20abjbgjj":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<59100]
        xbounds = np.array([59017, 59021, 59026,59028, 59030, 59032, 59034,
                            59036, 59038, 59040, 59045, 59047, 59050, 59052,
                            59055, 59062, 59067, 59082])
    elif name == "ZTF18abfcmjw":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58660]
        xbounds = np.arange(texp, texp+4, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58583.3, 58584, 58585, 58586, 58587, 58588,
                                       58589, 58591, 58591.3, 58595, 58597, 58599,
                                       58604, 58610, 58614, 58620])])
    elif name == "ZTF20aburywx":
        tb = load_ipac_lc(name, baseline_end = 59075, basecorr = True,
                          doplot = False, dobin = False)
        tb = tb[tb["mjd"]<59140]
        xbounds = np.arange(texp, texp+7, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([59086, 59095, 59102, 59110, 59120])])
    elif name == "ZTF18abvkwla":
        tb = load_ipac_lc(name)
        tb = tb[tb["mjd"]<58420]
        xbounds = np.array([58372, 58373, 58374, 58375, 58376, 58377, 58378, 58379,
                            58380, 58381, 58384, 58385, 58386])
    elif name == "ZTF20acigmel":
        #tb = load_ipac_lc(name, load_other=False)
        #tb = tb[tb["mjd"]<59200]
        #tb = tb[tb["mjd"]>59125]
        #tb = Table.from_pandas(tb)
        df = asci.read("../data/data_20xnd/Perley2021_table1.dat")
        #df = df[df["col2"]!="P48+ZTF"]
        ixr = (df["col3"]=="r") | (df["col3"]=="R") 
        dfr = df[ixr]
        ixg = (df["col3"]=="g") 
        dfg = df[ixg]
        dfg["F"] = "g"
        dfr["F"] = "r"
        df = vstack([dfg, dfr])
        df.rename_column("col1", "mjd")
        df.rename_column("col2", "Telescope")
        df["uJy"] = 10**(-0.4* df["col4"]) * 3631e+6
        df["duJy"] = df["col5"] * np.log(10) / 2.5 * df["uJy"]
        df["Fratio"] = df["uJy"].data / (3631e+6)
        df["Fratio_unc"] = df["duJy"].data / (3631e+6)   
        df.remove_columns(["col3", "col4", "col5", "col6"])
        tb = df[np.argsort(df["mjd"].data)]
        #tb = vstack([tb, df])
        #tb = tb[np.argsort(tb["mjd"].data)]
        tb = tb.to_pandas()
        xbounds = np.array([59134, 59135, 59136, 59137, 59138, 59139.1,
                            59139.5, 59140.4, 59141.5, 59142.5, 59142.9, 59141.5,
                            59143.5, 59145, 59145.3, 59146, 59148, 59156.5, 59157.5, 59158.5, 
                            59163, 59168])
    elif name == "ZTF21aakilyd":
        f = open("../data/data_21csp/photometry.txt", "r")
        lines = f.readlines()
        f.close()
        lines = lines[6:-1]
        
        mjds = [float(x[6:16]) for x in lines]
        Fs = [x[17] for x in lines]
        mags = [float(x[23:28]) for x in lines]
        emags = [float(x[29:33]) for x in lines]
        df = Table(data = [mjds, Fs, mags, emags], names = ["mjd", "F", "mag", "emag"])
        #df = df[df["col2"]!="P48+ZTF"]
        ixr = (df["F"]=="r")
        ixg = (df["F"]=="g") 
        ix = ixr | ixg
        df =df[ix]
        df["uJy"] = 10**(-0.4* df["mag"]) * 3631e+6
        df["duJy"] = df["emag"] * np.log(10) / 2.5 * df["uJy"]
        df["Fratio"] = df["uJy"].data / (3631e+6)
        df["Fratio_unc"] = df["duJy"].data / (3631e+6)   
        tb = df[np.argsort(df["mjd"].data)]
        #tb = vstack([tb, df])
        #tb = tb[np.argsort(tb["mjd"].data)]
        tb = tb.to_pandas()
        xbounds = np.array([59256,59257,59258,59259,59260,59261,59263,
                            59265, 59267, 59269, 59271, 59273, 59280,
                            59284.5, 59285.5, 59286.5, 59289, 59290.3, 59292,
                            59294, 59298])
    elif name == "ZTF19aayejww":
        f = open("../data/data_19hgp/phot.dat","r")
        lines = f.readlines()
        f.close()
        lines = np.array(lines)
        inds = []
        for x in lines:
            inds.append( "\x06" in x)
        inds = np.array(inds)
        lines = lines[inds]
        mjds = [float(x.split(" ")[0]) for x in lines]
        mags = [float(x.split(" ")[2].replace(":", ".")) for x in lines]
        emags = [float(x.split(" ")[4].replace(":", ".")) for x in lines]
        Fs = [x.split(" ")[6].split("\n")[0] for x in lines]
        df = Table(data = [mjds, Fs, mags, emags], names = ["mjd", "F", "mag", "emag"])
        ixr = (df["F"]=="r")
        ixg = (df["F"]=="g") 
        ix = ixr | ixg
        df =df[ix]
        df["uJy"] = 10**(-0.4* df["mag"]) * 3631e+6
        df["duJy"] = df["emag"] * np.log(10) / 2.5 * df["uJy"]
        df["Fratio"] = df["uJy"].data / (3631e+6)
        df["Fratio_unc"] = df["duJy"].data / (3631e+6)   
        tb = df[np.argsort(df["mjd"].data)]
        tb = tb[tb["mjd"]<58700]
        #tb = vstack([tb, df])
        #tb = tb[np.argsort(tb["mjd"].data)]
        tb = tb.to_pandas()
        xbounds = np.arange(texp, texp+21, 1)
        xbounds = np.hstack([xbounds, 
                             np.array([58663, 58664, 58667, 58668, 58670, 58672, 58675, 58678])])
    print ("explosion mjd = %.1f, z = %.4f"%(texp, z))
    
    texp = texp_
    tb["phase"] = (tb["mjd"] - texp) / (1+z)
    tb = tb[tb["phase"]>0]
    if xbounds is not None:
        xbounds = xbounds[np.argsort(xbounds)]
    return tb, xbounds


def add_source(ax, name):
    df = asci.read("./color_data/ho2021a_gold.dat")
    c_unc_max = 1
    
    if name == "ZTF20abfhyil":
        z = 0.1353
        texp = 59012
        sptype = "cow"
        atname = "AT2020mrf"
        ebv = 0.0174
    else:
        ind = df["ztfname"] == name
        z = float(df["z"].data[ind])
        texp = float(df["texp"].data[ind])
        sptype = df["sptype"].data[ind]
        atname = df["atname"].data[ind][0]
        ebv = df["ebv"].data[ind][0]
    
    tb, xbounds = load_lc(name, z, texp)

    Rg = 3.655730969791824 # in ZTF g band, Ag = Rg * E(B-V)
    Rr = 2.602659967686094 # in ZTF r band, Ar = Rr * E(B-V)
    Egmr = Rg*ebv - Rr*ebv
    ctb = calc_color(tb, xbounds = xbounds)
    ctb = ctb[ctb["gmr_unc"] < c_unc_max]
    ctb["phase"] = (ctb["mjd"] - texp) / (1+z)
    ctb["phase_unc"] = ctb["mjd_unc"] / (1+z)
    ctb["gmr"] -= Egmr
    
    color = "k"
    linestyle = "-"
    marker = "."
    mksize = 6
    linewidth = 0.6
    zorder = 1
    if name == "ZTF20abfhyil":
        color = "#ff0000"
        marker = "*"
        mksize = 10
        linewidth = 1.2
        zorder = 10
    elif name == "ZTF18abcfcoo":
        color = "k"
        marker = "D"
        mksize  = 4
        linewidth = 1.0
        zorder = 4
    elif name == 'ZTF18abvkwla':
        color = "#17d459"
        marker = "P"
        mksize  = 6
        linewidth = 1.0
        zorder = 8
    elif name == "ZTF20acigmel":
        color = "#6200ff"
        marker = "o"
        mksize  = 5
        linewidth = 1.0
        zorder = 6
    
    elif sptype == "Icn":
        linestyle = "--"
        linewidth = 0.75
        colormap1 = plt.cm.tab20c
        colormap2 = plt.cm.tab20b
        if name =="ZTF21aakilyd":
            color = colormap2(0/19)
        if name == 'ZTF19aayejww':
            color = colormap2(1/19)
    
    elif sptype == "Ibn":
        linestyle = ":"
        linewidth = 0.8
        colormap1 = plt.cm.tab20c
        colormap2 = plt.cm.tab20b
        if name == "ZTF19aapfmki":
            color = colormap1(8/19)
        if name == "ZTF19abobxik":
            color = colormap1(9/19)
        if name == "ZTF19abuvqgw":
            color = colormap1(10/19)
        if name == "ZTF19acayojs":
            color = colormap1(11/19)
        if name == "ZTF18aakuewf":
            color = colormap2(5/19)
        if name == "ZTF19aakssbm":
            color = colormap2(6/19)
    elif name == "ZTF19abyjzvd":
        linestyle = "-"
        linewidth = 0.7
        colormap2 = plt.cm.tab20b
        color = colormap2(4/19)
    
    elif sptype == "IIn":
        color = "mediumaquamarine"
        linestyle = "-."   
        linewidth = 0.7
    
    elif sptype == "II":
        linestyle = ":"
        linewidth = 0.7
        colormap2 = plt.cm.tab20b
        if name == "ZTF20aazchcq":
            color = colormap2(16/19)
        if name == "ZTF20abjbgjj":
            color = colormap2(17/19)
        if name == "ZTF20acigusw":
            color = colormap2(18/19)
        if name == "ZTF20aayrobw":
            color = colormap2(19/19)
    elif sptype == "Ic-BL":
        colormap1 = plt.cm.tab20c
        color = colormap1(0/19)
    elif sptype == "Ic":
        linestyle = "--"
        linewidth = 0.75
        colormap1 = plt.cm.tab20c
        color = colormap1(2/19)
        
    
    elif sptype == "IIb":
        colormap1 = plt.cm.tab20c
        colormap2 = plt.cm.tab20b
        linestyle = ":"
        linewidth = 0.8
        if name == "ZTF18abwkrbl":
            color = colormap1(4/19)
        if name == "ZTF19accjfgv":
            color = "chocolate"
            color = colormap1(5/19)
        if name == "ZTF20aahfqpm":
            color = "darkred"
            color = colormap1(6/19)
        if name == "ZTF20aaxhzhc":
            color = "salmon"
            color = colormap1(7/19)
        if name == "ZTF20aburywx":
            color = colormap2(14/19)
        if name == "ZTF20aclfmwn":
            color = colormap2(15/19)
    elif sptype == "Ib":
        colormap1 = plt.cm.tab20c
        colormap2 = plt.cm.tab20b
        linestyle = "-"
        linewidth = 0.7
        if name == "ZTF18abfcmjw":
            #color = colormap1(1/19)
            color = colormap2(12/19)
        if name == "ZTF18abvkmgw":
            #color = colormap1(3/19)
            color = colormap2(13/19)
        
    
    mytext = atname[4:]
    ax.errorbar(ctb["phase"], ctb["gmr"], ctb["gmr_unc"], 
                xerr = ctb["phase_unc"], fmt = marker, color = color,
                linestyle = linestyle,
                markersize = mksize, elinewidth = 0.6, linewidth = linewidth,
                label = mytext, zorder=zorder)
    
