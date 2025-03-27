def stats_helper(y,x):
    res=stats.theilslopes(y,x)
    if (not np.all(x==y)) and (not np.nanmax(np.diff(x))==0) and (not np.isnan(np.nanmax(np.diff(x)))):
        res2=stats.linregress(x,y)
        return np.array([res.low_slope,res.slope,res.high_slope,res2.pvalue])
    else:
        return np.array([res.low_slope,res.slope,res.high_slope,0])

# MAIN LOOP TO COMPUTE THE REGRESSION
conv = True # WE ARE USING CONVERGENCE
plat = 40 # southern latitude bound
for lat_n in [60]:
    print(lat_n)
    for c,case in enumerate(list(names.keys())):
            for e in range(Sflux_all[case].ens.size):
                for j in range(0,AMOC[case].isel(ens=e).size-30,1):
                    if conv:
                        # all fluxes defined positive northward, convergence is south-north 
                        var_in = -(Sflux_1000_all[case].sel(lat=lat_n).isel(ens=e,year=slice(j,j+30)) \
                                  -Sflux_1000_all[case].isel(ens=e,year=slice(j,j+30)))/1E6 #convergence
                    else:
                        var_in = Sflux_1000_all[case].isel(ens=e,year=slice(j,j+30))/1E6 #flux
                    #
                    res=xr.apply_ufunc(stats_helper,AMOC[case].isel(ens=e,year=slice(j,j+30))/1E9,
                                       var_in,
                                       input_core_dims=[['year'],['year']],
                                       output_core_dims=[['dum']],
                                       vectorize=True,
                                       dask='parallelized',
                                       dask_gufunc_kwargs={'output_sizes':{'dum':4}}
                    ).compute()
                    if j==0:
                        res_out=res.expand_dims({'year':np.array([j])})
                    else:
                        res_out=xr.concat([res_out,res.expand_dims({'year':np.array([j])})],dim='year')
                #
                #find max AMOC-Sflux slope                
                jj_pos=[]
                jj_neg=[]
                jj_abs=[]
                for l in range(res.lat.size):
                    if np.all(np.isfinite(res_out.isel(dum=1,lat=l))):
                        jj_pos.append(np.where(res_out.isel(dum=1,lat=l)==max(res_out.isel(dum=1,lat=l)))[0][0])
                        jj_neg.append(np.where(res_out.isel(dum=1,lat=l)==min(res_out.isel(dum=1,lat=l)))[0][0])
                        jj_abs.append(np.where(abs(res_out.isel(dum=1,lat=l))==max(abs(res_out.isel(dum=1,lat=l))))[0][0])
                    else:
                        jj_pos.append(0)
                        jj_neg.append(0)
                        jj_abs.append(0)
                # pick up values at the time of the max AMOC-Sflux slope
                jj_pos=xr.DataArray(np.array(jj_pos),dims='max_points')
                ll_pos=xr.DataArray(range(0,res_out.lat.size),dims='max_points')
                jj_neg=xr.DataArray(np.array(jj_neg),dims='min_points')
                ll_neg=xr.DataArray(range(0,res_out.lat.size),dims='min_points')
                jj_abs=xr.DataArray(np.array(jj_abs),dims='abs_max_points')
                ll_abs=xr.DataArray(range(0,res_out.lat.size),dims='abs_max_points')
                min_dAMOCdt=np.where(AMOC[case].isel(ens=e).rolling(year=30).mean().differentiate('year')==AMOC[case].isel(ens=e).rolling(year=30).mean().differentiate('year').min())[0][0]
                res_out_pos=res_out.isel(year=jj_pos,lat=ll_pos).expand_dims({'ens':np.array([e])})
                res_out_neg=res_out.isel(year=jj_neg,lat=ll_neg).expand_dims({'ens':np.array([e])})
                res_out_abs=res_out.isel(year=jj_abs,lat=ll_abs).expand_dims({'ens':np.array([e])})
                res_out_min=res_out.isel(year=min_dAMOCdt-30).expand_dims({'ens':np.array([e])})
                # average across ensembles
                if e==0:
                    res_all_pos=res_out_pos
                    res_all_neg=res_out_neg
                    res_all_abs=res_out_abs
                    res_all_min=res_out_min
                else:
                    res_all_pos=xr.concat([res_all_pos,res_out_pos],dim='ens')
                    res_all_neg=xr.concat([res_all_neg,res_out_neg],dim='ens')
                    res_all_abs=xr.concat([res_all_abs,res_out_abs],dim='ens')
                    res_all_min=xr.concat([res_all_min,res_out_min],dim='ens')
            #
            if c==0:
                AMOC_Sflux_slope_pos=res_all_pos.expand_dims({'case':np.array([case])})
                AMOC_Sflux_slope_neg=res_all_neg.expand_dims({'case':np.array([case])})
                AMOC_Sflux_slope_abs=res_all_abs.expand_dims({'case':np.array([case])})
                AMOC_Sflux_slope_min=res_all_min.expand_dims({'case':np.array([case])})
                AMOC_stable=AMOC[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E9
                if conv:
                    Sflux_stable = -(Sflux_1000_all[case].sel(lat=lat_n).isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)) \
                                  -Sflux_1000_all[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size))).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E6
                else:
                    Sflux_stable = Sflux_1000_all[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E6 
            elif case not in ['1pct']:
                AMOC_Sflux_slope_pos=xr.concat([AMOC_Sflux_slope_pos,res_all_pos.expand_dims({'case':np.array([case])})],dim='case')
                AMOC_Sflux_slope_neg=xr.concat([AMOC_Sflux_slope_neg,res_all_neg.expand_dims({'case':np.array([case])})],dim='case')
                AMOC_Sflux_slope_abs=xr.concat([AMOC_Sflux_slope_abs,res_all_abs.expand_dims({'case':np.array([case])})],dim='case')
                AMOC_Sflux_slope_min=xr.concat([AMOC_Sflux_slope_min,res_all_min.expand_dims({'case':np.array([case])})],dim='case')
                AMOC_stable=xr.concat([AMOC_stable,AMOC[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E9],dim='case')
                if conv:
                    Sflux_stable_dum=(Sflux_1000_all[case].sel(lat=lat_n).isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)) \
                                      -Sflux_1000_all[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size))).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E6
                else:
                    Sflux_stable_dum=Sflux_1000_all[case].isel(year=slice(AMOC[case].year.size-30,AMOC[case].year.size)).expand_dims({'case':np.array([case])}).mean(dim=('year','ens'))/1E6
                #
                Sflux_stable = xr.concat([Sflux_stable, Sflux_stable_dum],dim='case')
    
    res_stable=xr.apply_ufunc(stats_helper,AMOC_stable,Sflux_stable,
                       input_core_dims=[['case'],['case']],
                       output_core_dims=[['dum']],
                       vectorize=True,
                       dask='parallelized',
                       dask_gufunc_kwargs={'output_sizes':{'dum':4}}).compute()
    ################################
    # FIGURE 4
    plat=40
    lat_n=60
    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(10,10),sharex=True,sharey=True)
    for c,case in enumerate(names.keys()):
        print(case)
        if conv:
            var_in = -(Sflux_1000_all[case].sel(lat=lat_n) \
                                      -Sflux_1000_all[case].sel(lat=plat))/1E6
        else:
            var_in = Sflux_1000_all[case].sel(lat=plat)/1E6
        if c==0:
            var_in_all = var_in.isel(year=slice(70,100)).mean('ens').expand_dims('case')
            AMOC_all   = AMOC[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E9
        elif case not in ['1pct']:
            var_in_all = xr.concat([var_in_all,var_in.isel(year=slice(70,100)).mean('ens').expand_dims('case')],dim='case')
            AMOC_all   = xr.concat([AMOC_all,AMOC[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E9],dim='case')
        #
        axes.flatten()[c].plot(var_in.stack(z=('year','ens')),AMOC[case].stack(z=('year','ens'))/1E9,
                               '.',color='C'+str(c),label=case)
        for nn,n in enumerate(range(0,AMOC[case].year.size-30,10)):
            for e in range(Sflux[case].ens.size):
                 THEIL=stats.theilslopes(AMOC[case].isel(ens=e).squeeze().values[n:n+30]/1E9,var_in.isel(ens=e).squeeze().values[n:n+30])
                 axes.flatten()[c].plot(var_in.isel(ens=e).squeeze().values[n:n+30],
                                        THEIL.slope*var_in.isel(ens=e).squeeze().values[n:n+30]+THEIL.intercept,
                                        color='k',lw=3)
                 axes.flatten()[c].plot(var_in.isel(ens=e).squeeze().values[n:n+30],
                                        THEIL.slope*var_in.isel(ens=e).squeeze().values[n:n+30]+THEIL.intercept,
                                        color='C'+str(c),lw=1)
        axes.flatten()[c].set_title(case)
        if case in ['1pct']:
            THEIL     = stats.theilslopes(AMOC[case].isel(ens=e).squeeze()/1E9,var_in.isel(ens=e).squeeze())
            THEIL_all = stats.theilslopes(AMOC_all.stack(z=('year','case')).squeeze(),var_in_all.stack(z=('year','case')).squeeze())
            xaxis_dum = np.linspace(np.round(var_in_all.min().values,decimals=1),np.round(var_in_all.max().values,decimals=1))
            for ax in axes.flatten():
                ax.plot(xaxis_dum,xaxis_dum*THEIL.slope+THEIL.intercept,color='gray',ls='--',lw=2,zorder=0)
                ax.plot(xaxis_dum,xaxis_dum*THEIL_all.slope+THEIL_all.intercept,color='k',ls='--',lw=2,zorder=0)
    #
    if conv:
        Sflux_conv_ora  = -(ORAS5_1000.Sflux_60N - ORAS5_1000['Sflux_'+str(plat)+'N'])/1E6
        Sflux_conv_ecco = -(ECCO_data.Sflux_60N_1000.squeeze()-ECCO_data['Sflux_'+str(plat)+'N_1000'].squeeze())
    else:
        Sflux_conv_ora  = ORAS5_1000['Sflux_'+str(plat)+'N']/1E6
        Sflux_conv_ecco = ECCO_data['Sflux_'+str(plat)+'N_1000'].squeeze()
    #
    res_ecco = stats.theilslopes(AMOC['ecco'].squeeze(),Sflux_conv_ecco)
    res_oras5 = stats.theilslopes(AMOC['ORAS5'].squeeze(),Sflux_conv_ora)
    axes.flatten()[-1].plot(Sflux_conv_ecco,AMOC['ecco'].squeeze(),
                           '.',color='k',label='ECCO')
    axes.flatten()[-1].plot(Sflux_conv_ecco,res_ecco.slope*Sflux_conv_ecco+res_ecco.intercept,
                           '-',color='k')
    axes.flatten()[-1].plot(Sflux_conv_ora.squeeze(),AMOC['ORAS5'].squeeze(),
                            '.',color='r',label='ORAS5')
    axes.flatten()[-1].plot(Sflux_conv_ora.squeeze(),res_oras5.slope*Sflux_conv_ora.squeeze()+res_oras5.intercept,
                            '-',color='r')
    axes.flatten()[-1].legend()
    #
    extra_artists=[]
    for a,ax in enumerate(axes.flatten()):
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[a],transform=ax.transAxes, fontsize=20)
        ax.axvline(x=0,color='gray',lw=0.5)
        extra_artists.append(txt1)
    #
    xlab=fig.text(0.05,0.5,'AMOC [Sv]',rotation='vertical',ha='center',va='center',fontsize=18)
    fig.subplots_adjust(wspace=0.05)
    if conv:
        ylab=fig.text(0.5,0.05,r'Salt Transport Convergence [g/kg $\cdot$ Sv]',ha='center',va='center',fontsize=18)
        extra_artists.extend([xlab,ylab])
        fig.savefig('../Figures/Sflux_1000_convergence_AMOC_scatter_slopes_annual_lat'+str(plat)+'.png',dpi=300,
                    bbox_inches='tight',bbox_extra_artists=extra_artists)
    else:
        ylab=fig.text(0.5,0.05,r'Salt Transport [g/kg $\cdot$ Sv]',ha='center',va='center',fontsize=18)
        extra_artists.extend([xlab,ylab])
        fig.savefig('../Figures/Sflux_1000_AMOC_scatter_slopes_annual'+str(plat)+'.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    #
    # ###################################################################
    #
    # FIGURE 5: SALT TRANSPORT CONVERGENCE SLOPE AT 40 AND BETWEEN 34-60
    co2_axis=np.array([1.0,1.4,1.6,1.8,2,2.8,4])
    fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
    if False:
        AMOC_Sflux_slope_abs.assign_coords(abs_max_points=AMOC_Sflux_slope_abs.lat).sel(abs_max_points=slice(-40,60)).\
            mean('ens').assign_coords(case=np.array([1.0,1.4,1.6,1.8,2.0,2.8,4.0])).rename({'case':'CO2'}).isel(dum=1).squeeze().\
            plot(ax=ax2,vmin=-100/1E3,vmax=100/1E3,cmap=plt.get_cmap('RdBu_r'))
    else:
        AMOC_Sflux_slope_min.mean('ens').assign_coords(case=np.array([1.0,1.4,1.6,1.8,2.0,2.8,4.0])).rename({'case':'CO2'}).isel(dum=1).squeeze(). \
            plot(ax=ax2,vmin=-100/1E3,vmax=100/1E3,cmap=plt.get_cmap('RdBu_r'))
    ax2.set_yticks(co2_axis)
    ax2.axvline(x=lat_n,color='gray',lw=3)
    # points where slope does not have the same sign
    jj,ii=np.where(abs(np.sign(AMOC_Sflux_slope_min.isel(dum=slice(0,2))).sum('dum')).min('ens')<3)
    jj,ii=np.where(AMOC_Sflux_slope_min.isel(dum=-1).max('ens')>0.05)
    ax2.plot(AMOC_Sflux_slope_min.lat[ii],np.array([1.0,1.4,1.6,1.8,2.0,2.8,4.0])[jj],'o',color='gray')
    ax2.set_ylabel(r'CO$_2$',fontsize=20)
    ax2.set_xlabel(r'Latitude [$\degree$N]', fontsize=20)
    ax2.set_xticks([-20,0,20,plat,40,60])
    ax2.set_xlim(-34,60)
    if conv:
        ax2.set_title(r'AMOC - S$_{transport}$ convergence slope',fontsize=20)
    else:
        ax2.set_title(r'AMOC - S$_{transport}$ slope',fontsize=20)
    #
    if False:
        ll=np.where(AMOC_Sflux_slope_abs.lat==plat)[0][0]
        for s, sm in enumerate(co2_axis):
            ax1.errorbar(co2_axis[s],AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).mean('ens'),
                        yerr=np.array([AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).mean('ens')-AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).min('ens'),
                                       AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1,case=s).max('ens')-AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1,case=s).mean('ens')])[:,np.newaxis],
                        color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1).mean('ens'),ls='-',marker='o',color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=0).mean('ens'),ls='--',color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=2).mean('ens'),ls='--',color='k')
    else:
        for s, sm in enumerate(co2_axis):
            ax1.errorbar(co2_axis[s],AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens'),
                        yerr=np.array([AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens')-AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).min('ens'),
                                       AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).max('ens')-AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens')])[:,np.newaxis],
                        color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=1).mean('ens'),ls='-',marker='o',color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=0).mean('ens'),ls='--',color='k')
        ax1.plot(co2_axis,AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=2).mean('ens'),ls='--',color='k')
    
    ax1.plot(np.array(IpctCO2_axis),res_out.sel(lat=plat).isel(dum=1),color='C1',label='1pct')
    ax1.fill_between(np.array(IpctCO2_axis),res_out.sel(lat=plat).isel(dum=0),res_out.sel(lat=plat).isel(dum=2),color='C1',alpha=0.25)
    # ECCO RESULTS
    Sflux_conv_ecco = -(ECCO_data.Sflux_60N_1000.squeeze()-ECCO_data['Sflux_'+str(plat)+'N_1000'].squeeze())
    medslope_ecco,medint_ecco,lo_slope_ecco,up_slope_ecco = stats.theilslopes(AMOC['ecco'].squeeze(),Sflux_conv_ecco)
    medslope_ecco1,medint_ecco1,lo_slope_ecco1,up_slope_ecco1 = stats.theilslopes(AMOC['ecco'].squeeze()[:15],Sflux_conv_ecco[:15])
    medslope_ecco2,medint_ecco2,lo_slope_ecco2,up_slope_ecco2 = stats.theilslopes(AMOC['ecco'].squeeze()[-15:],Sflux_conv_ecco[-15:])
    ax1.errorbar(np.array([np.mean(CO2_values[13:39])]),medslope_ecco,yerr=np.array([medslope_ecco-lo_slope_ecco,
                                                                                    up_slope_ecco-medslope_ecco])[:,np.newaxis],
                fmt='d',color='C3',label='ECCO')
    ax1.errorbar(np.array([np.mean(CO2_values[13:39][:15])]),medslope_ecco1,
                yerr=np.array([medslope_ecco1-lo_slope_ecco1,up_slope_ecco1-medslope_ecco1])[:,np.newaxis],fmt='o',color='C3')
    ax1.errorbar(np.array([np.mean(CO2_values[13:39][-15:])]),medslope_ecco2,
                yerr=np.array([medslope_ecco2-lo_slope_ecco2,up_slope_ecco2-medslope_ecco2])[:,np.newaxis],fmt='o',color='C3')
    #ORAS5
    Sflux_conv_ora = - (ORAS5_1000.Sflux_60N - ORAS5_1000['Sflux_'+str(plat)+'N']).squeeze()/1E6
    medslope_ora,medint_ora,lo_slope_ora,up_slope_ora = stats.theilslopes(AMOC['ORAS5'].squeeze(),Sflux_conv_ora)
    ax1.errorbar(np.array([np.mean(CO2_values[:40])]),medslope_ora,
                yerr=np.array([medslope_ora-lo_slope_ora,up_slope_ora-medslope_ora])[:,np.newaxis],
                fmt='d',color='C4',label='ORAS5')
    for jj,j in enumerate([5,15,25]):
        medslope_ora1,medint_ora1,lo_slope_ora1,up_slope_ora1 = stats.theilslopes(AMOC['ORAS5'].squeeze()[j:j+15],Sflux_conv_ora[j:j+15])
        ax1.errorbar(np.array([np.mean(CO2_values[j:j+15])]),medslope_ora1,
                yerr=np.array([medslope_ora1-lo_slope_ora1,up_slope_ora1-medslope_ora1])[:,np.newaxis],fmt='o',color='C4')
    #
    ax1.axhline(y=THEIL_all.slope,color='gray',ls='-',lw=2,zorder=0)
    ax1.fill_between(np.array([0,5]),THEIL_all.low_slope,THEIL_all.high_slope,color='gray',alpha=0.25,lw=2,zorder=0)
    ax1.set_xlim(0.99,4.01)
    ax1.set_ylim(-25/1E3,75/1E3)
    ax1.legend()
    ax1.set_ylabel(r'AMOC - S$_{transport}$ convergence slope [Sv / ( g/kg $\times$ Sv)]',fontsize=20)
    ax1.set_xlabel('CO2 forcing',fontsize=20)
    extra_artists=[]
    for a,ax in enumerate([ax1,ax2]):
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[a],transform=ax.transAxes, fontsize=20)
        extra_artists.append(txt1)

    fig.subplots_adjust(wspace=0.1)
    fig.savefig('../Figures/AMOC_Sconv_1000_'+str(plat)+'N_60N_slope_and_convergence.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    # ######################################
    # FIGURE S9: SALT TRANSPORT CONVERGENCE 40N-60N AMOC REGRESSION AS A FUNCTION OF SPG RHO
    # #####################################
    rho_abs=True
    rho_control=butils.eosben07_sig0(0,SST_SPG['x1.0'],SSS_SPG['x1.0'],pref=0).mean().values
    sigma0_control=rho_control-1000
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10),sharex=True,sharey=True)
    if rho_abs:
        rho_axis=np.array(rho_max)
    else:
        rho_axis=-100*(np.array(rho_max)-rho_control)/(rho_control-1000)
    #
    ll=np.where(AMOC_Sflux_slope_abs.lat==plat)[0][0]
    for s, sm in enumerate(rho_axis):
        if False:
            ax.errorbar(rho_axis[s],AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).mean('ens'),
                    yerr=np.array([AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).mean('ens')-AMOC_Sflux_slope_abs.isel(case=s,abs_max_points=ll,dum=1).min('ens'),
                                   AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1,case=s).max('ens')-AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1,case=s).mean('ens')])[:,np.newaxis],
                    color='k')
        else:
            ax.errorbar(rho_axis[s],AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens'),
                        yerr=np.array([AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens')-AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).min('ens'),
                                       AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).max('ens')-AMOC_Sflux_slope_min.sel(lat=plat).isel(case=s,dum=1).mean('ens')])[:,np.newaxis],
                        color='k')
    for j in [[0,1,2],[3,4,5,6]]:
        if False:
            ax.plot(rho_axis[j],AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=1).mean('ens')[j],ls='-',marker='o',color='k')
            ax.plot(rho_axis[j],AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=0).mean('ens')[j],ls='--',color='k')
            ax.plot(rho_axis[j],AMOC_Sflux_slope_abs.isel(abs_max_points=ll,dum=2).mean('ens')[j],ls='--',color='k')
        else:
            ax.plot(rho_axis[j],AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=1).mean('ens')[j],ls='-',marker='o',color='k')
            ax.plot(rho_axis[j],AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=0).mean('ens')[j],ls='--',color='k')
            ax.plot(rho_axis[j],AMOC_Sflux_slope_min.sel(lat=plat).isel(dum=2).mean('ens')[j],ls='--',color='k')
    # 1% results
    rho_1pct=butils.eosben07_sig0(0,SST_SPG['1pct'],SSS_SPG['1pct']).rolling(year=30,center=True).mean().mean('ens')
    if rho_abs:
        rho_axis_1pct=rho_1pct
    else:
        rho_axis_1pct=-100*(rho_1pct-rho_control)/(rho_control-1000)
    #
    ax.plot(rho_axis_1pct[15:-15],res_out.sel(lat=plat).isel(dum=1),color='C1',label='1pct')
    ax.fill_between(rho_axis_1pct[15:-15],res_out.sel(lat=plat).isel(dum=0),res_out.sel(lat=plat).isel(dum=2),color='C1',alpha=0.25)
    # ECCO RESULTS
    if rho_abs:
        sigma0_ecco         = butils.eosben07_sig0(0,SST_SPG['ecco'],SSS_SPG['ecco'],pref=0)
        rho_axis_ecco_mean  = np.array([np.nanmean(sigma0_ecco.values)])
        rho_axis_ecco1      = np.array([np.nanmean(sigma0_ecco.values[:15])])
        rho_axis_ecco2      = np.array([np.nanmean(sigma0_ecco.values[-15:])])
    else:
        sigma0_ecco=butils.eosben07_sig0(0,SST_SPG['ecco'],SSS_SPG['ecco'],pref=0)-1000
        scale_factor        = -100/sigma0_control
        rho_axis_ecco_mean  = scale_factor*np.array([(np.nanmean(sigma0_ecco.values)-sigma0_control)])
        rho_axis_ecco1      = scale_factor*np.array([(np.nanmean(sigma0_ecco.values[:15])-sigma0_control)])
        rho_axis_ecco2      = scale_factor*np.array([(np.nanmean(sigma0_ecco.values[-15:])-sigma0_control)])
    #
    Sflux_conv_ecco = -(ECCO_data.Sflux_60N_1000.squeeze()-ECCO_data['Sflux_'+str(plat)+'N_1000'].squeeze())
    medslope_ecco,medint_ecco,lo_slope_ecco,up_slope_ecco = stats.theilslopes(AMOC['ecco'].squeeze(),Sflux_conv_ecco)
    medslope_ecco1,medint_ecco1,lo_slope_ecco1,up_slope_ecco1 = stats.theilslopes(AMOC['ecco'].squeeze()[:15],Sflux_conv_ecco[:15])
    medslope_ecco2,medint_ecco2,lo_slope_ecco2,up_slope_ecco2 = stats.theilslopes(AMOC['ecco'].squeeze()[-15:],Sflux_conv_ecco[-15:])
    ax.errorbar(rho_axis_ecco_mean,medslope_ecco,yerr=np.array([medslope_ecco-lo_slope_ecco,up_slope_ecco-medslope_ecco])[:,np.newaxis],
                fmt='d',color='C3',label='ECCO')
    ax.errorbar(rho_axis_ecco1,medslope_ecco1,
                yerr=np.array([medslope_ecco1-lo_slope_ecco1,up_slope_ecco1-medslope_ecco1])[:,np.newaxis],fmt='o',color='C3')
    ax.errorbar(rho_axis_ecco2,medslope_ecco2,
                yerr=np.array([medslope_ecco2-lo_slope_ecco2,up_slope_ecco2-medslope_ecco2])[:,np.newaxis],fmt='o',color='C3')
    #ORAS5
    Sflux_conv_ora = -(ORAS5_1000.Sflux_60N - ORAS5_1000['Sflux_'+str(plat)+'N']).squeeze()/1E6
    if rho_abs:
        sigma0_ora=butils.eosben07_sig0(0,SST_SPG['ORAS5'],SSS_SPG['ORAS5'],pref=0)
        rho_axis_oras5_mean = np.array([np.nanmean(sigma0_ora.values)])
    else:
        sigma0_ora=butils.eosben07_sig0(0,SST_SPG['ORAS5'],SSS_SPG['ORAS5'],pref=0)-1000
        rho_axis_oras5_mean=scale_factor*np.array([(np.mean(sigma0_ora.values)-sigma0_control)])#
    
    medslope_ora,medint_ora,lo_slope_ora,up_slope_ora = stats.theilslopes(AMOC['ORAS5'].squeeze(),Sflux_conv_ora)
    ax.errorbar(rho_axis_oras5_mean,medslope_ora,
                yerr=np.array([medslope_ora-lo_slope_ora,up_slope_ora-medslope_ora])[:,np.newaxis],
                fmt='d',color='C4',label='ORAS5')
    for jj,j in enumerate([5,15,25]):
        medslope_ora1,medint_ora1,lo_slope_ora1,up_slope_ora1 = stats.theilslopes(AMOC['ORAS5'].squeeze()[j:j+15],Sflux_conv_ora[j:j+15])
        if rho_abs:
            rho_axis_oras5=np.array([np.mean(sigma0_ora[j:j+15])])
        else:
            rho_axis_oras5=-100*np.array([(np.mean(sigma0_ora[j:j+15])-sigma0_control)])/sigma0_control
        ax.errorbar(rho_axis_oras5,medslope_ora1,
                yerr=np.array([medslope_ora1-lo_slope_ora1,up_slope_ora1-medslope_ora1])[:,np.newaxis],fmt='o',color='C4')
    
    ax.axhline(y=THEIL_all.slope,color='gray',ls='-',lw=2,zorder=0)
    ax.fill_between(np.array([np.min(rho_axis_1pct),np.max(rho_axis_1pct)]),THEIL_all.low_slope,THEIL_all.high_slope,color='gray',alpha=0.25,lw=2,zorder=0)
    ax.set_ylim(-25/1E3,75/1E3)
    ax.set_ylabel(r'AMOC - S$_{transport}$ convergence slope [Sv / ( g/kg $\times$ Sv)]',fontsize=20)
    ax.legend()
    if rho_abs:
        ax.set_xlim(np.nanmax([np.nanmax(rho_axis),np.nanmax(rho_axis_1pct[15:-15])]),np.nanmin([np.nanmin(rho_axis),np.nanmin(rho_axis_1pct[15:-15])]))
        ax.set_xlabel(r'SPG potential density [kg m$^{-3}$]',fontsize=20)
        fig.savefig('../Figures/AMOC_Sconv_1000_'+str(plat)+'N_60N_slope_and_convergence_as_a_function_of_SPG_rho_abs.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    else:
        ax.set_xlim(np.nanmin([np.nanmin(rho_axis),np.nanmin(rho_axis_1pct[15:-15])]),np.nanmax([np.nanmax(rho_axis),np.nanmax(rho_axis_1pct[15:-15])]))
        ax.set_xlabel('Percentage reduction in SPG potential density anomaly [%]',fontsize=20)
        fig.savefig('../Figures/AMOC_Sconv_1000_'+str(plat)+'N_60N_slope_and_convergence_as_a_function_of_SPG_rho.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
