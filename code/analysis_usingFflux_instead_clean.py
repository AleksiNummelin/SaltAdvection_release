import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#
# regression slopes
#
for s in range(4):
    stable_Fflux=[]
    stable_AMOC=[]
    stable_rho=[]
    stable_drhodS=[]
    r_all={}
    s_med=[]
    s_up=[]
    s_lo=[]
    rho_max=[]
    drhodS_max=[]
    lags=np.arange(-20,21)
    for c,case in enumerate(names.keys()):
        if case not in ['1pct']:
            r_ens=[]
            s_m=[]
            s_u=[]
            s_l=[]
            rho_d=[]
            drhodS_d=[]
            for e in range(Fflux[case].ens.size):
                r=[]
                ll = np.where(np.isfinite(Fflux[case].isel(ens=e,S0=s)))[0]
                med_slope=[]
                lo_slope=[]
                up_slope=[]
                rho_dum=[]
                drhodS_dum=[]
                for j in range(0,AMOC[case].isel(ens=e).size-20,10):
                    mslope,medint,l_slope,u_slope=stats.theilslopes(AMOC[case].isel(ens=e).values[j:j+30]/1E9,Fflux[case].isel(ens=e,S0=s).values[j:j+30]/1E6)
                    dum=butils.eosben07_sig0(0,SST_SPG[case].isel(ens=e).values[j:j+30],SSS_SPG[case].isel(ens=e).values[j:j+30],pref=0)
                    res=stats.theilslopes(dum,SSS_SPG[case].isel(ens=e).values[j:j+30])
                    drhodS_dum.append(res.slope)
                    med_slope.append(mslope)
                    lo_slope.append(l_slope)
                    up_slope.append(u_slope)
                    rho_dum.append(dum)
                #find max AMOC-Sflux slope
                jj=np.where(abs(np.array(med_slope))==max(abs(np.array(med_slope))))[0][0]
                stable_Fflux.append(Fflux[case].isel(ens=e,S0=s).values[ll[-30:]]/1E6)
                stable_AMOC.append(AMOC[case].isel(ens=e).values[ll[-30:]]/1E9)
                dum=butils.eosben07_sig0(0,SST_SPG[case].isel(ens=e).values[ll[-30:]],SSS_SPG[case].isel(ens=e).values[ll[-30:]],pref=0)
                res=stats.theilslopes(dum,SSS_SPG[case].isel(ens=e).values[ll[-30:]])
                stable_rho.append(dum)
                stable_drhodS.append(res.slope)
                # pick up values at the time of the max AMOC-Sflux slope
                s_m.append(med_slope[jj])
                s_u.append(up_slope[jj])
                s_l.append(lo_slope[jj])
                rho_d.append(rho_dum[jj]) #
                drhodS_d.append(drhodS_dum[jj])
                for lag in lags:
                    if lag==0:
                        rdum,p=stats.pearsonr(Fflux[case].isel(ens=e,S0=s).values[ll]/1E6,AMOC[case].isel(ens=e).values[ll]/1E9)
                    elif lag<0:
                        rdum,p=stats.pearsonr(Fflux[case].isel(ens=e,S0=s).values[ll][-lag:]/1E6,AMOC[case].isel(ens=e).values[ll][:lag]/1E9)
                    else:
                        rdum,p=stats.pearsonr(Fflux[case].isel(ens=e,S0=s).values[ll][:-lag]/1E6,AMOC[case].isel(ens=e).values[ll][lag:]/1E9)
                    r.append(rdum)
                r_ens.append(np.array(r))
            # average across ensembles
            s_med.append(np.mean(s_m))
            s_up.append(np.mean(s_u))
            s_lo.append(np.mean(s_l))
            rho_max.append(np.mean(rho_d))
            drhodS_max.append(np.mean(drhodS_d))
            if Fflux[case].ens.size>1:
                r_all[case]=np.mean(np.array(r_ens),0)
            else:
                r_all[case]=np.array(r_ens)
    #
    medslope_stable,medint_stable,lo_slope_stable,up_slope_stable=stats.theilslopes(np.array(stable_AMOC),np.array(stable_Fflux))
    #1 pct
    case='1pct'
    s_med_1pct=[]
    s_up_1pct=[]
    s_lo_1pct=[]
    r=[]
    years=np.arange(0,AMOC[case].size-30,1).astype(int)
    for n in years:
        medslope,medint,lo_slope,up_slope=stats.theilslopes(AMOC[case].squeeze().values[n:n+30]/1E9,Fflux[case].isel(S0=s).squeeze().values[n:n+30]/1E6)
        s_med_1pct.append(medslope)
        s_up_1pct.append(up_slope)
        s_lo_1pct.append(lo_slope)
    for lag in lags:
        if lag==0:
            rdum,p=stats.pearsonr(Fflux[case].isel(ens=0,S0=s).values/1E6,AMOC[case].isel(ens=0).values/1E9)
        elif lag<0:
            rdum,p=stats.pearsonr(Fflux[case].isel(ens=0,S0=s).values[-lag:]/1E6,AMOC[case].isel(ens=0).values[:lag]/1E9)
        else:
            rdum,p=stats.pearsonr(Fflux[case].isel(ens=0,S0=s).values[:-lag]/1E6,AMOC[case].isel(ens=0).values[lag:]/1E9)
        r.append(rdum)
    #
    r_all[case]=np.array(r)
    #
    #################################################################
    #
    # FIGURES S3-S4: FRESHWATER TRANSPORT - AMOC LAGGED CORRELATION 
    #
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10),sharex=True,sharey=True)
    for c,case in enumerate(names.keys()):
        ax.plot(lags,np.array(r_all[case]).squeeze()**2,label=case)
    ax.legend()
    ax.axvline(x=0,color='gray',lw=2,ls='--')
    ax.set_ylabel(r'Pearson r$^2$',fontsize=20)
    ax.set_xlabel('Lag [years]',fontsize=20)
    fig.savefig('../Figures/LAGGED_correlation_Fflux_S0_'+str(s)+'.png',dpi=300)
    #
    # FIGURES S1-S2: FRESHWATER TRANSPORT - AMOC REGRESSION
    #
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10),sharex=True,sharey=True)
    ax.plot(np.array([1.0,1.4,1.6,1.8,2,2.8,4]),np.array(s_med),ls='-',marker='o',color='k')
    ax.errorbar(np.array(IpctCO2_axis),np.array(s_med_1pct),yerr=np.array([np.array(s_med_1pct)-np.array(s_lo_1pct),np.array(s_up_1pct)-np.array(s_med_1pct)]),
                fmt='o',color='C1',label='1pct')
    ax.plot(np.array(IpctCO2_axis),np.array(s_med_1pct),color='C1')
    ax.axhline(y=medslope_stable,color='gray',ls='-')
    ax.fill_between(np.arange(1,5),lo_slope_stable,up_slope_stable,color='gray',alpha=0.5)
    #
    ax.set_xlim(1,4)
    ax.legend()
    ax.set_ylabel(r'AMOC - F$^{34\degree S}_{transport}$  (Sv / Sv)',fontsize=20)
    ax.set_xlabel('CO2 forcing',fontsize=20)
    fig.savefig('../Figures/AMOC_Ftransport_slope_S0_'+str(s)+'_v2.png',dpi=300)
