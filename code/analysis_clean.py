import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import dask, distributed
import os
import socket
from scipy import stats,signal
import sys
sys.path.append('/nird/home/anu074/code/ECCOv4-py/ECCOv4-py')
import ecco_v4_py as ecco
import gsw
import BLOM_utils as butils
import string
from pathlib import Path

if __name__ == '__main__':
    local_dir = '/scratch/anu074/'+socket.gethostname()+'/'
    if not os.path.isdir(local_dir):
        os.system('mkdir -p '+local_dir)
        print('created folder '+local_dir)
    #
    n_workers = 4
    n_threads = 2
    processes = True
    cluster   = distributed.LocalCluster(n_workers=n_workers,threads_per_worker=n_threads,processes=processes,
                                              local_directory=local_dir,lifetime='48 hour',lifetime_stagger='10 minutes',
                                              lifetime_restart=True,dashboard_address=None,worker_dashboard_address=None)
    client    = distributed.Client(cluster)
    #
    # FOR COMPLETENES THE CODE TO COMPUTE THE DIFFERENT VARIABLES IS INCLUDED
    # BUT THE FULL DATA IS NOT SHARED SO THE FLAGS BELOW NEED TO REMAIN 'False'
    compute_data=False 
    compute_ecco=False
    # CO2 DATA FROM
    # https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt
    CO2_glob=np.array([
        1979,336.85,0.11,
        1980,338.91,0.07,
        1981,340.11,0.09,
        1982,340.86,0.03,
        1983,342.53,0.06,
        1984,344.07,0.08,
        1985,345.54,0.07,
        1986,346.97,0.07,
        1987,348.68,0.10,
        1988,351.16,0.07,
        1989,352.78,0.07,
        1990,354.05,0.07,
        1991,355.39,0.07,
        1992,356.09,0.06,
        1993,356.83,0.07,
        1994,358.33,0.08,
        1995,360.17,0.05,
        1996,361.93,0.04,
        1997,363.05,0.05,
        1998,365.70,0.04,
        1999,367.80,0.05,
        2000,368.96,0.06,
        2001,370.57,0.05,
        2002,372.59,0.04,
        2003,375.15,0.04,
        2004,376.95,0.06,
        2005,378.98,0.05,
        2006,381.15,0.05,
        2007,382.90,0.04,
        2008,385.02,0.05,
        2009,386.50,0.04,
        2010,388.76,0.06,
        2011,390.63,0.05,
        2012,392.65,0.06,
        2013,395.40,0.06,
        2014,397.34,0.05,
        2015,399.65,0.05,
        2016,403.06,0.06,
        2017,405.22,0.07,
        2018,407.61,0.07,
        2019,410.07,0.07,
        2020,412.44,0.06,
        2021,414.70,0.07,
        2022,417.07,0.07,
        2023,419.31,0.15
    ])
    #
    CO2_years=CO2_glob[0::3]
    CO2_values=CO2_glob[1::3]/285
    # DEFINE 1PCT INCREASE
    IpctCO2=[285]
    for j in range(1,165):
        IpctCO2.append(IpctCO2[-1]+0.01*IpctCO2[-1])
    IpctCO2=np.array(IpctCO2)/285
    IpctCO2_axis=[]
    for j in range(0,165-30,1):
        IpctCO2_axis.append(np.mean(IpctCO2[j:j+30]))
    #
    # DEFINE THE MAIN LOGIC FOR HOLDING THE DATA
    path1='/projects/NS2345K/noresm/cases/'
    path2='/projects/NS9560K/noresm/cases/'
    path3='/projects/NS9560K/users/adagj/cases/'
    path4='/projects/NS10013K/noresm/cases/'
    names={
        'x1.0':{'name':['N1850_f19_tn14_20190621','N1850_f19_tn14_20190621','N1850_f19_tn14_20190722'],'path':[path2,path2,path2],'nyears':1200},
        'x1.4': {'name':['NCO2x1.4_f19_tn14_20210914','NCO2x1.4_N1850_f19_tn14_20210317b_1705_20220430',
                        'NCO2x1.4_N1850_f19_tn14_20210317b_1870_20220430'],
                 'path':[path1,path1,path1],'nyears':1200},
        'x1.6':{'name':['NCO2x1.6_f19_tn14_20240910'],'path':[path4],'nyears':1200},
        'x1.8':{'name':['NCO2x1.8_f19_tn14_20240910'],'path':[path4],'nyears':1200},
        'x2.0': {'name':[['NCO2x2_f19_tn14_20210819'],['NCO2x2_N1850_f19_tn14_20210317b_1705_20220430','NCO2x2_N1850_f19_tn14_20210317b_1705_20240909'],
                         ['NCO2x2_N1850_f19_tn14_20210317b_1870_20220430','NCO2x2_N1850_f19_tn14_20210317b_1870_20240909']],
                 'path':[[path1],[path1,path4],[path1,path4]],'nyears':1200,'nyears2':[[1200],[480,1200],[600,1200]]},
        'x2.8': {'name':['NCO2x2.8_f19_tn14_20210914'],'path':[path1],'nyears':1200},
        'x4.0': {'name':['NCO2x4_f19_tn14_20190624','NCO2x4_f19_tn14_20210819',
                         #'NCO2x4frc2_f19_tn14_20210429',
                         'NCO2x4frc2_CPLHIST_f19_tn14_20230327'],
                 'path':[path2,path2,
                         #path2,
                         path3],'nyears':1200},
        '1pct':{'name':[['N1PCT_f19_tn14_20190626','N1PCT_f19_tn14_20190712']],'path':[[path2,path2]],'nyears':1980,'nyears2':[[1440,600]]}
    }
    # BELOW WE DEFINE SOME BLOM (NORESM2-LM) SPECIFIC METRICS
    if compute_data:
        grid_blom=xr.open_dataset('/projects/NS9869K/noresm/topo_beta/grid_tnx1v4_20170622.nc')
        mask=np.logical_and(np.logical_and(np.logical_and(grid_blom.plat>50,
                                                      grid_blom.plat<60),
                                       np.logical_and(grid_blom.plon>-50,
                                                      grid_blom.plon<-40)),
                        grid_blom.pdepth>2000)
        jinds,iinds=np.where(mask)
        jinds=xr.DataArray(jinds,dims='points')
        iinds=xr.DataArray(iinds,dims='points')
        iinds_34S = xr.DataArray(np.arange(55,130).astype(int),dims='points')
        jinds_34S =	xr.DataArray(104*np.ones(iinds_34S.size).astype(int),dims='points')
        #
        mertra     = butils.read_mertra('../../mertra_index_tnx1v4_20190615.dat')
        basin_mask = xr.open_dataset('../../ocean_regions_tnx1v4_20190729.nc')
        lat        = np.arange(-34,61)
        mertra_atlantic = {}
        for l in lat:
            l_mask = basin_mask.region.isel(x=xr.DataArray(mertra[l][:,0],dims='points'),y=xr.DataArray(mertra[l][:,1],dims='points'))
            linds=np.where(l_mask==2)[0]
            mertra_atlantic[l] = mertra[l][linds,:]
    #
    # DEFINE A FEW DICTS THAT WILL HOLD THE DATA
    Sflux_1000_all = {}
    AMOC           = {}
    Sflux          = {}
    Sflux_all      = {}
    Fflux          = {}
    SST_SPG        = {}
    SSS_SPG        = {}
    S0=xr.DataArray([34.6,34.7,34.8,35.],dims='S0')
    #
    # THE MAIN LOOP TO COMPUTE THE MAIN VARIABLES (NOT USED HERE)
    # OR TO DOWNLOAD THE DATA.
    for case in list(names.keys()):
        if compute_data:
            print(case)
            for n,name in enumerate(names[case]['name'][:3]):
                print(n,name)
                nyears=names[case]['nyears']
                if (case in ['x1.0']) and (n==1):
                    filelist=sorted(glob.glob(names[case]['path'][n]+name+'/ocn/hist/'+name+'.*.hm.*-*.nc'))[n*nyears:(n+1)*nyears]
                elif (case in ['x2.0','1pct']):
                    filelist=[]
                    for nn,fpath in enumerate(names[case]['path'][n]):
                        filelist.extend(sorted(glob.glob(fpath+name[nn]+'/ocn/hist/'+name[nn]+'.*.hm.*-*.nc'))[:names[case]['nyears2'][n][nn]])
                    filelist=filelist[:nyears]
                else:
                    filelist=sorted(glob.glob(names[case]['path'][n]+name+'/ocn/hist/'+name+'.*.hm.*-*.nc'))[:nyears]
                dtime=12*50
                for jj,j in enumerate(range(0,min(nyears,len(filelist)),dtime)):
                    #print(jj,j)
                    data_dum=xr.open_mfdataset(filelist[j:j+dtime],combine='nested',concat_dim='time',chunks={'time':120})
                    #JFM mean SST and SSS
                    SSTdum=((grid_blom.parea*data_dum.sst).isel(y=jinds,x=iinds).sum('points')/ \
                            grid_blom.parea.isel(y=jinds,x=iinds).sum('points')).resample(time='QS').mean(dim='time').isel(time=slice(0,-1,4)). \
                            expand_dims({'ens':np.array([n])})#.compute()
                    SSSdum=((grid_blom.parea*data_dum.sss).isel(y=jinds,x=iinds).sum('points')/ \
                            grid_blom.parea.isel(y=jinds,x=iinds).sum('points')).resample(time='QS').mean(dim='time').isel(time=slice(0,-1,4)). \
                            expand_dims({'ens':np.array([n])})#.compute()
                    # annual mean AMOC and Sflux
                    amoc_dum=data_dum['mmflxd'].isel(region=0,depth=slice(26,-1)).sel(lat=26). \
                        max(dim=['depth']).groupby('time.year').mean().expand_dims({'ens':np.array([n])})#.compute()
                    sflux_dum=data_dum.msflx.isel(region=0).sel(lat=-34).groupby('time.year').mean().expand_dims({'ens':np.array([n])})#.compute()
                    sflux_all=data_dum.msflx.isel(region=0).groupby('time.year').mean().expand_dims({'ens':np.array([n])})#.compute()
                    # annual mean salt transport in upper 1000 m
                    for ll,l in enumerate(lat):
                        print(ll,l)
                        iinds_lat = xr.DataArray(mertra_atlantic[l][:,0].astype(int),dims='points')
                        jinds_lat = xr.DataArray(mertra_atlantic[l][:,1].astype(int),dims='points')
                        if ll==0:
                            sflx1000_lat = (data_dum.usflxlvl.isel(depth=slice(0,37),x=iinds_lat,y=jinds_lat)*mertra_atlantic[l][:,2]+\
                                data_dum.vsflxlvl.isel(depth=slice(0,37),x=iinds_lat,y=jinds_lat)*mertra_atlantic[l][:,3]).sum(dim=('depth','points')). \
                                expand_dims({'ens':np.array([n]),'lat':np.array([l])})
                        else:
                            sflx1000_lat_dum = (data_dum.usflxlvl.isel(depth=slice(0,37),x=iinds_lat,y=jinds_lat)*mertra_atlantic[l][:,2]+\
                                data_dum.vsflxlvl.isel(depth=slice(0,37),x=iinds_lat,y=jinds_lat)*mertra_atlantic[l][:,3]).sum(dim=('depth','points')). \
                                expand_dims({'ens':np.array([n]),'lat':np.array([l])})
                            sflx1000_lat=xr.concat([sflx1000_lat,sflx1000_lat_dum],dim='lat')
                    # The previous loop is lazy and here we execute the computations. The graph is large,
                    # but it seems to be still quite efficient, taking about 10 min, for 50 year chunk.
                    # It could perhaps be possible to build a mapping from indiced to lat more directly, but not sure.
                    sflx1000_lat=sflx1000_lat.groupby('time.year').mean().compute()
                    #
                    vflx_zmean=data_dum.mmflxd.isel(region=0).sel(lat=-34)
                    S_zmean=(data_dum.salnlvl.isel(y=jinds_34S,x=iinds_34S)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points') \
                        /(data_dum.salnlvl.isel(y=jinds_34S,x=iinds_34S).notnull().astype(float)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points')
                    fflux_dum=-(1/S0)*(vflx_zmean.fillna(0)*(S_zmean.fillna(0)-S0)).integrate('depth').groupby('time.year').mean().expand_dims({'ens':np.array([n])})#.compute()
                    #amoc_dum=amoc_dum.assign_coords({'year':range(amoc_dum.year.size)})
                    if jj==0 and n==0:
                        AMOC[case]             = amoc_dum
                        Sflux[case]            = sflux_dum
                        Sflux_all[case]        = sflux_all
                        Sflux_1000_all[case]   = sflx1000_lat
                        SST_SPG[case]          = SSTdum
                        SSS_SPG[case]          = SSSdum
                        Fflux[case]            = fflux_dum
                    elif jj>0 and n==0:
                        AMOC[case]    = xr.concat([AMOC[case],amoc_dum],dim='year')
                        Sflux[case]   = xr.concat([Sflux[case],sflux_dum],dim='year')
                        Sflux_all[case]   = xr.concat([Sflux_all[case],sflux_all],dim='year')
                        Sflux_1000_all[case]   = xr.concat([Sflux_1000_all[case],sflx1000_lat],dim='year')
                        SST_SPG[case] = xr.concat([SST_SPG[case],SSTdum],dim='time')
                        SSS_SPG[case] = xr.concat([SSS_SPG[case],SSSdum],dim='time')
                        Fflux[case]   = xr.concat([Fflux[case],fflux_dum],dim='year')
                    elif jj==0 and n>0:
                        amoc_2   = amoc_dum
                        sflux_2  = sflux_dum
                        sflux_all_2 = sflux_all
                        sflx1000_lat_2 = sflx1000_lat
                        SST_SPG2 = SSTdum
                        SSS_SPG2 = SSSdum
                        fflux_2  = fflux_dum
                    elif jj>0 and n>0:
                        amoc_2   = xr.concat([amoc_2,amoc_dum],dim='year')
                        sflux_2  = xr.concat([sflux_2,sflux_dum],dim='year')
                        sflux_all_2  = xr.concat([sflux_all_2,sflux_all],dim='year')
                        sflx1000_lat_2 = xr.concat([sflx1000_lat_2,sflx1000_lat],dim='year')
                        SST_SPG2 = xr.concat([SST_SPG2,SSTdum],dim='time')
                        SSS_SPG2 = xr.concat([SSS_SPG2,SSSdum],dim='time')
                        fflux_2  = xr.concat([fflux_2,fflux_dum],dim='year')
                    data_dum.close()
                if n==0:
                    AMOC[case]  = AMOC[case].assign_coords({'year':range(AMOC[case].year.size)})
                    Sflux[case] = Sflux[case].assign_coords({'year':range(Sflux[case].year.size)})
                    Sflux_all[case] = Sflux_all[case].assign_coords({'year':range(Sflux_all[case].year.size)})
                    Sflux_1000_all[case] = Sflux_1000_all[case].assign_coords({'year':range(Sflux_1000_all[case].year.size)})
                    SST_SPG[case] = SST_SPG[case].assign_coords({'time':range(AMOC[case].year.size)}).rename({'time':'year'})
                    SSS_SPG[case] = SSS_SPG[case].assign_coords({'time':range(AMOC[case].year.size)}).rename({'time':'year'})
                    Fflux[case]   = Fflux[case].assign_coords({'year':range(Fflux[case].year.size)})
                elif n>0:
                    AMOC[case]  = xr.concat([AMOC[case],amoc_2.assign_coords({'year':range(amoc_2.year.size)})],dim='ens')
                    Sflux[case] = xr.concat([Sflux[case],sflux_2.assign_coords({'year':range(sflux_2.year.size)})],dim='ens')
                    Sflux_all[case] = xr.concat([Sflux_all[case],sflux_all_2.assign_coords({'year':range(sflux_all_2.year.size)})],dim='ens')
                    Sflux_1000_all[case] = xr.concat([Sflux_1000_all[case],sflx1000_lat_2.assign_coords({'year':range(sflx1000_lat_2.year.size)})],dim='ens')
                    SST_SPG[case] = xr.concat([SST_SPG[case], SST_SPG2.assign_coords({'time':range(amoc_2.year.size)}).rename({'time':'year'})],dim='ens')
                    SSS_SPG[case] = xr.concat([SSS_SPG[case], SSS_SPG2.assign_coords({'time':range(amoc_2.year.size)}).rename({'time':'year'})],dim='ens')
                    Fflux[case]   =	xr.concat([Fflux[case], fflux_2.assign_coords({'year':range(fflux_2.year.size)})],dim='ens')
            # SAVE DATA AS YOU GO
            Sflux_1000_all[case].to_dataset(name='Sflux_1000_all').to_netcdf('/projects/NS9874K/AMOC/Sflux_1000_all_'+case+'.nc')
            # could build logic around if not Path(target).is_file() i.e if file exists or not
            AMOC[case].to_dataset(name='AMOC').to_netcdf('/projects/NS9874K/AMOC/AMOC_'+case+'.nc')
            Sflux[case].to_dataset(name='Sflux').to_netcdf('/projects/NS9874K/AMOC/Sflux_'+case+'.nc')
            Sflux_all[case].to_dataset(name='Sflux_all').to_netcdf('/projects/NS9874K/AMOC/Sflux_all_'+case+'.nc')
            #Sflux_1000_all[case].to_dataset(name='Sflux_1000_all').to_netcdf('/projects/NS9874K/AMOC/Sflux_1000_all_'+case+'.nc')
            SST_SPG[case].to_dataset(name='SST_SPG').to_netcdf('/projects/NS9874K/AMOC/SST_SPG_'+case+'.nc')
            SSS_SPG[case].to_dataset(name='SSS_SPG').to_netcdf('/projects/NS9874K/AMOC/SSS_SPG_'+case+'.nc')
            Fflux[case].to_dataset(name='Fflux').to_netcdf('/projects/NS9874K/AMOC/Fflux_'+case+'.nc')
        else:
            # LOAD DATA INSTEAD
            AMOC[case]      = xr.open_dataset('../data/NorESM2-LM/AMOC_'+case+'.nc').AMOC.load()
            Sflux[case]     = xr.open_dataset('../data/NorESM2-LM/Sflux_'+case+'.nc').Sflux.load()
            Sflux_all[case] = xr.open_dataset('../data/NorESM2-LM/Sflux_all_'+case+'.nc').Sflux_all.load()
            Sflux_1000_all[case] = xr.open_dataset('../data/NorESM2-LM/Sflux_1000_all_'+case+'.nc').Sflux_1000_all.load()
            SST_SPG[case]   = xr.open_dataset('../data/NorESM2-LM/SST_SPG_'+case+'.nc').SST_SPG.load()
            SSS_SPG[case]   = xr.open_dataset('../data/NorESM2-LM/SSS_SPG_'+case+'.nc').SSS_SPG.load()
            Fflux[case]     = xr.open_dataset('../data/NorESM2-LM/Fflux_'+case+'.nc').Fflux.load()

    if compute_ecco:
        path_ecco='/projects/NS9874K/anu074/ECCO/'
        var={'vol':'ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4',
             'sal':'ECCO_L4_OCEAN_3D_SALINITY_FLUX_LLC0090GRID_MONTHLY_V4R4'
             }
        data={}
        grid=xr.open_dataset(path_ecco+'GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc')
        for name in list(var.keys()):
           #
            data[name]  = xr.open_mfdataset(glob.glob(path_ecco+var[name]+'/*'))
        #
        ds_ecco  = xr.merge([data['vol'],grid])
        ds2_ecco = xr.merge([data['sal'],grid])
        #
        AMOC['ecco']   = ecco.calc_meridional_stf(ds_ecco,lat_vals=26,basin_name='atlExt')
        Sflux['ecco']  = ecco.calc_meridional_salt_trsp(ds2_ecco,lat_vals=-34,basin_name='atlExt')
        Sflux['ecco_26N']  = ecco.calc_meridional_salt_trsp(ds2_ecco,lat_vals=26,basin_name='atlExt')
        Sflux['ecco_29N']  = ecco.calc_meridional_salt_trsp(ds2_ecco,lat_vals=29,basin_name='atlExt')
        Sflux['ecco_40N']  = ecco.calc_meridional_salt_trsp(ds2_ecco,lat_vals=40,basin_name='atlExt')
        Sflux['ecco_60N']  = ecco.calc_meridional_salt_trsp(ds2_ecco,lat_vals=60,basin_name='atlExt')
        #
        AMOC['ecco']   = AMOC['ecco'].moc.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_int']  = Sflux['ecco'].salt_trsp.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_26N_int']  = Sflux['ecco_26N'].salt_trsp.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_29N_int']  = Sflux['ecco_29N'].salt_trsp.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_40N_int']  = Sflux['ecco_40N'].salt_trsp.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_60N_int']  = Sflux['ecco_60N'].salt_trsp.squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_1000']      = Sflux['ecco'].salt_trsp_z.isel(k=slice(0,28)).sum('k').squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_26N_1000']  = Sflux['ecco_26N'].salt_trsp_z.isel(k=slice(0,28)).sum('k').squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_29N_1000']  = Sflux['ecco_29N'].salt_trsp_z.isel(k=slice(0,28)).sum('k').squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_40N_1000']  = Sflux['ecco_40N'].salt_trsp_z.isel(k=slice(0,28)).sum('k').squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        Sflux['ecco_60N_1000']  = Sflux['ecco_60N'].salt_trsp_z.isel(k=slice(0,28)).sum('k').squeeze().groupby('time.year').mean().expand_dims({'ens':np.array([0])})
        ECCO_data = xr.merge([AMOC['ecco'].to_dataset(name='AMOC').drop_vars('lat'),
                              Sflux['ecco_int'].to_dataset(name='Sflux').drop_vars('lat'),
                              Sflux['ecco_26N_int'].to_dataset(name='Sflux_26N').drop_vars('lat'),
                              Sflux['ecco_29N_int'].to_dataset(name='Sflux_29N').drop_vars('lat'),
                              Sflux['ecco_40N_int'].to_dataset(name='Sflux_40N').drop_vars('lat'),
                              Sflux['ecco_60N_int'].to_dataset(name='Sflux_60N').drop_vars('lat'),
                              Sflux['ecco_1000'].to_dataset(name='Sflux_1000').drop_vars('lat'),
                              Sflux['ecco_26N_1000'].to_dataset(name='Sflux_26N_1000').drop_vars('lat'),
                              Sflux['ecco_29N_1000'].to_dataset(name='Sflux_29N_1000').drop_vars('lat'),
                              Sflux['ecco_40N_1000'].to_dataset(name='Sflux_40N_1000').drop_vars('lat'),
                              Sflux['ecco_60N_1000'].to_dataset(name='Sflux_60N_1000').drop_vars('lat'),
        ])
        ECCO_data.to_netcdf('/projects/NS9874K/anu074/ECCO_AMOC_Sflux_new2.nc')
        if True:
            ECCO_T_S=xr.open_mfdataset(sorted(glob.glob(path_ecco+'/ECCO_L4_TEMP_SALINITY_LLC0090GRID_MONTHLY_V4R4/')))
            ecco_SPG_mask=np.logical_and(np.logical_and(np.logical_and(grid.YC>50,
                                                      grid.YC<60),
                                           np.logical_and(grid.XC>-50,
                                                         grid.XC<-40)),
                            grid.Depth>2000)
            tinds,jinds,iinds=np.where(ecco_SPG_mask)
            tinds=xr.DataArray(tinds,dims='points')
            jinds=xr.DataArray(jinds,dims='points')
            iinds=xr.DataArray(iinds,dims='points')
            SPG_ECCO_T=ECCO_T_S.THETA.isel(k=0,tile=tinds,j=jinds,i=iinds).resample(time='QS').mean(dim='time').isel(time=slice(0,-1,4))
            SPG_ECCO_S=ECCO_T_S.SALT.isel(k=0,tile=tinds,j=jinds,i=iinds).resample(time='QS').mean(dim='time').isel(time=slice(0,-1,4))
            SPG_ECCO_A=grid.rA.isel(tile=tinds,j=jinds,i=iinds)
            T_out=(SPG_ECCO_T*SPG_ECCO_A).sum('points')/SPG_ECCO_A.sum('points')
            S_out=(SPG_ECCO_S*SPG_ECCO_A).sum('points')/SPG_ECCO_A.sum('points')
            xr.merge([T_out.to_dataset(name='SST_SPG'),S_out.to_dataset(name='SSS_SPG')]).to_netcdf('/projects/NS9874K/anu074/ECCO_SST_SSS_SPG_JFM.nc')
    #######################
    # Load the ECCO data
    ECCO_data = xr.open_dataset('../data/ECCO/ECCO_AMOC_Sflux_new2.nc')
    AMOC['ecco'] = ECCO_data.AMOC
    Sflux['ecco'] = ECCO_data.Sflux
    SST_SPG['ecco'] = xr.open_dataset('../data/ECCO/ECCO_SST_SSS_SPG_JFM.nc').SST_SPG
    SSS_SPG['ecco'] = xr.open_dataset('../data/ECCO/ECCO_SST_SSS_SPG_JFM.nc').SSS_SPG
    # Load the ORAS5 data
    ORAS5 = xr.open_dataset('../data/ORAS5/AMOC_Sflux_1979_2018.nc')
    #
    ORAS5_1000 = xr.open_dataset('../data/ORAS5/AMOC_Sflux_0_1000_1979_2018.nc')
    #
    Sflux['ORAS5'] = ORAS5.Sflux.expand_dims({'ens':np.array([0])})/1E6
    AMOC['ORAS5']  = ORAS5.AMOC.expand_dims({'ens':np.array([0])})/1E6
    SST_SPG['ORAS5'] = xr.open_dataset('../data/ORAS5/SPG_SSS_SST_1979_2018_JFM.nc').SST_SPG.isel(deptht=0)
    SSS_SPG['ORAS5'] = xr.open_dataset('../data/ORAS5/SPG_SSS_SST_1979_2018_JFM.nc').SSS_SPG.isel(deptht=0)
    # ALL DATA IS NOW LOADED
    # ----------------------
    #
    # ------------------------------
    # THE MAIN LOOP FOR COMPUTING THE 
    # REGRESSION SLOPES
    #
    stable_Sflux=[]
    stable_AMOC=[]
    stable_rho=[]
    stable_drhodS=[]
    r_all={}
    s_med=[]
    s_up=[]
    s_lo=[]
    rho_max=[]
    drhodS_max=[]
    jj_inds={}
    lags=np.arange(-20,21)
    for c,case in enumerate(names.keys()):
        if case not in ['1pct']:
            r_ens=[]
            s_m=[]
            s_u=[]
            s_l=[]
            rho_d=[]
            drhodS_d=[]
            for e in range(Sflux[case].ens.size):
                r=[]
                ll = np.where(np.isfinite(Sflux[case].isel(ens=e)))[0]
                med_slope=[]
                lo_slope=[]
                up_slope=[]
                rho_dum=[]
                drhodS_dum=[]
                years=np.arange(0,AMOC[case].isel(ens=e).size-30,1).astype(int)
                #
                for j in years:
                    mslope,medint,l_slope,u_slope=stats.theilslopes(AMOC[case].isel(ens=e).values[j:j+30]/1E9,
                                                                    Sflux[case].isel(ens=e).values[j:j+30]/1E6)
                    dum=butils.eosben07_sig0(0,SST_SPG[case].isel(ens=e).values[j:j+30],SSS_SPG[case].isel(ens=e).values[j:j+30],pref=0)
                    res=stats.theilslopes(dum,SSS_SPG[case].isel(ens=e).values[j:j+30])
                    drhodS_dum.append(res.slope)
                    med_slope.append(mslope)
                    lo_slope.append(l_slope)
                    up_slope.append(u_slope)
                    rho_dum.append(dum)
                #find max AMOC-Sflux slope
                jj=np.where(abs(np.array(med_slope))==max(abs(np.array(med_slope))))[0][0]
                jj_inds[case+'_'+str(e)]=slice(years[jj],
                                               years[jj]+30)
                stable_Sflux.append(Sflux[case].isel(ens=e).values[ll[-30:]]/1E6)
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
                drhodS_d.append(drhodS_dum[jj]) #but would it make sense to pick up the max instead here?
                for lag in lags:
                    if lag==0:
                        rdum,p=stats.pearsonr(Sflux[case].isel(ens=e).values[ll]/1E6,AMOC[case].isel(ens=e).values[ll]/1E9)
                    elif lag<0:
                        rdum,p=stats.pearsonr(Sflux[case].isel(ens=e).values[ll][-lag:]/1E6,AMOC[case].isel(ens=e).values[ll][:lag]/1E9)
                    else:
                        rdum,p=stats.pearsonr(Sflux[case].isel(ens=e).values[ll][:-lag]/1E6,AMOC[case].isel(ens=e).values[ll][lag:]/1E9)
                    r.append(rdum)
                r_ens.append(np.array(r))
            # average across ensembles
            s_med.append(s_m) #.append(np.mean(s_m))
            s_up.append(np.mean(s_u))
            s_lo.append(np.mean(s_l))
            rho_max.append(np.mean(rho_d))
            drhodS_max.append(np.mean(drhodS_d))
            if Sflux[case].ens.size>1:
                r_all[case]=np.mean(np.array(r_ens),0)
            else:
                r_all[case]=np.array(r_ens)
    #
    medslope_stable,medint_stable,lo_slope_stable,up_slope_stable=stats.theilslopes(np.array(stable_AMOC),np.array(stable_Sflux))
    #1 pct
    case='1pct'
    s_med_1pct=[]
    s_up_1pct=[]
    s_lo_1pct=[]
    r=[]
    #
    years=np.arange(0,AMOC[case].size-30,1).astype(int)
    for n in years:
        medslope,medint,lo_slope,up_slope=stats.theilslopes(AMOC[case].squeeze().values[n:n+30]/1E9,
                                                            Sflux[case].squeeze().values[n:n+30]/1E6)
        s_med_1pct.append(medslope)
        s_up_1pct.append(up_slope)
        s_lo_1pct.append(lo_slope)
    for lag in lags:
        if lag==0:
            rdum,p=stats.pearsonr(Sflux[case].isel(ens=0).values/1E6,AMOC[case].isel(ens=0).values/1E9)
        elif lag<0:
            rdum,p=stats.pearsonr(Sflux[case].isel(ens=0).values[-lag:]/1E6,AMOC[case].isel(ens=0).values[:lag]/1E9)
        else:
            rdum,p=stats.pearsonr(Sflux[case].isel(ens=0).values[:-lag]/1E6,AMOC[case].isel(ens=0).values[lag:]/1E9)
        r.append(rdum)
    #
    r_all[case]=np.array(r)
    #
    #
    #####################
    # FIGURES BELOW HERE
    #
    # FIGURE 1: AMOC-SFLUX SCATTER WITH MAX SLOPE
    fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(10,10),sharex=True,sharey=True)
    for c,case in enumerate(names.keys()):
        print(case)
        axes.flatten()[c].plot(Sflux[case].stack(z=('year','ens'))/1E6,AMOC[case].stack(z=('year','ens'))/1E9,
                               '.',color='C'+str(c),label=case)
        if c==0:
            var_in_all = Sflux[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E6
            AMOC_all   = AMOC[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E9
        elif case not in ['1pct']:
            var_in_all = xr.concat([var_in_all,Sflux[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E6],dim='case')
            AMOC_all   = xr.concat([AMOC_all,AMOC[case].isel(year=slice(70,100)).mean('ens').expand_dims('case')/1E9],dim='case')
        #
        if case not in ['1pct']:
            for e in range(Sflux[case].ens.size):
                THEIL=stats.theilslopes(AMOC[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E9,\
                                    Sflux[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E6)
                axes.flatten()[c].plot(Sflux[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E6,
                                   THEIL.slope*Sflux[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E6+THEIL.intercept,
                                       color='k',lw=3)
                axes.flatten()[c].plot(Sflux[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E6,
                                       THEIL.slope*Sflux[case].isel(ens=e,year=jj_inds[case+'_'+str(e)]).values/1E6+THEIL.intercept,
                                       color='C'+str(c),lw=1)
        elif case in ['1pct']:
            THEIL     = stats.theilslopes(AMOC[case].squeeze()/1E9,Sflux[case].mean('ens').expand_dims('case')/1E6)
            THEIL_all = stats.theilslopes(AMOC_all.stack(z=('year','case')).squeeze(),var_in_all.stack(z=('year','case')).squeeze())
            xaxis_dum = np.linspace(np.round(var_in_all.min().values,decimals=1),np.round(var_in_all.max().values,decimals=1))
            for ax in axes.flatten():
                ax.plot(xaxis_dum,xaxis_dum*THEIL.slope+THEIL.intercept,color='gray',ls='--',lw=2,zorder=0)
                ax.plot(xaxis_dum,xaxis_dum*THEIL_all.slope+THEIL_all.intercept,color='k',ls='--',lw=2,zorder=0)
            for nn,n in enumerate(range(0,AMOC[case].year.size-30,10)):
                 THEIL=stats.theilslopes(AMOC[case].squeeze().values[n:n+30]/1E9,Sflux[case].squeeze().values[n:n+30]/1E6)
                 axes.flatten()[c].plot(Sflux[case].squeeze().values[n:n+30]/1E6,
                                        THEIL.slope*Sflux[case].squeeze().values[n:n+30]/1E6+THEIL.intercept,
                                        color='k',lw=3)
                 axes.flatten()[c].plot(Sflux[case].squeeze().values[n:n+30]/1E6,
                                        THEIL.slope*Sflux[case].squeeze().values[n:n+30]/1E6+THEIL.intercept,
                                        color='C'+str(c),lw=1)

        axes.flatten()[c].set_title(case)

    axes.flatten()[-1].plot(Sflux['ecco'].squeeze(),AMOC['ecco'].squeeze(),
                            '.',color='k',label='ECCO')
    axes.flatten()[-1].plot(Sflux['ORAS5'].squeeze(),AMOC['ORAS5'].squeeze(),
                            '.',color='r',label='ORAS5')
    axes.flatten()[-1].legend()
    xlab=fig.text(0.5,0.05,r'Salt Transport [g/kg $\cdot$ Sv]',ha='center',va='center',fontsize=18)
    ylab=fig.text(0.05,0.5,'AMOC [Sv]',rotation='vertical',ha='center',va='center',fontsize=18)
    extra_artists=[xlab,ylab]
    for a,ax in enumerate(axes.flatten()):
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[a],transform=ax.transAxes, fontsize=20)
        extra_artists.append(txt1)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig('../Figures/Sflux_AMOC_scatter_slopes_annual.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    #
    # Sflux - AMOC relation
    #################################################
    #
    # FIGURE 2: COMBINED AMOC-STRANSPORTR SLOPE AS A FUNCTION OF
    # CO2 AND DENSITY
    rho_abs=False
    rho_control=butils.eosben07_sig0(0,SST_SPG['x1.0'],SSS_SPG['x1.0'],pref=0).mean().values
    sigma0_control=rho_control-1000
    if rho_abs:
        rho_axis=np.array(rho_max)
    else:
        rho_axis=-100*(np.array(rho_max)-rho_control)/(rho_control-1000)
    #
    fig,(ax,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(20,10),sharex=False,sharey=True)
    s_med_mean=[]
    co2_axis=np.array([1.0,1.4,1.6,1.8,2,2.8,4])
    for s, sm in enumerate(s_med):
        s_med_mean.append(np.mean(sm))
        if len(sm)>1:
            ax.errorbar(co2_axis[s],np.mean(sm),yerr=np.array([np.mean(sm)-np.min(sm),np.max(sm)-np.mean(sm)])[:,np.newaxis],color='k')
            ax2.errorbar(rho_axis[s],np.mean(sm),yerr=np.array([np.mean(sm)-np.min(sm),np.max(sm)-np.mean(sm)])[:,np.newaxis],color='k')
    ax.plot(co2_axis,np.array(s_med_mean),ls='-',marker='o',color='k')
    ax.plot(co2_axis,np.array(s_lo),ls='--',color='k')
    ax.plot(co2_axis,np.array(s_up),ls='--',color='k')
    #
    ax.plot(np.array(IpctCO2_axis),np.array(s_med_1pct),color='C1',label='1pct')
    ax.fill_between(np.array(IpctCO2_axis),np.array(s_lo_1pct),np.array(s_up_1pct),color='C1',alpha=0.25)
    ax.axhline(y=medslope_stable,color='gray',ls='-')
    ax.fill_between(np.arange(1,5),lo_slope_stable,up_slope_stable,color='gray',alpha=0.25)
    # ECCO RESULTS
    medslope_ecco,medint_ecco,lo_slope_ecco,up_slope_ecco = stats.theilslopes(AMOC['ecco'].squeeze(),Sflux['ecco'].squeeze())
    medslope_ecco1,medint_ecco1,lo_slope_ecco1,up_slope_ecco1 = stats.theilslopes(AMOC['ecco'].squeeze()[:15],Sflux['ecco'].squeeze()[:15])
    medslope_ecco2,medint_ecco2,lo_slope_ecco2,up_slope_ecco2 = stats.theilslopes(AMOC['ecco'].squeeze()[-15:],Sflux['ecco'].squeeze()[-15:])
    ax.errorbar(np.array([np.mean(CO2_values[13:39])]),medslope_ecco,yerr=np.array([medslope_ecco-lo_slope_ecco,
                                                                                    up_slope_ecco-medslope_ecco])[:,np.newaxis],
                fmt='d',color='C3',label='ECCO')
    ax.errorbar(np.array([np.mean(CO2_values[13:39][:15])]),medslope_ecco1,
                yerr=np.array([medslope_ecco1-lo_slope_ecco1,up_slope_ecco1-medslope_ecco1])[:,np.newaxis],fmt='o',color='C3')
    ax.errorbar(np.array([np.mean(CO2_values[13:39][-15:])]),medslope_ecco2,
                yerr=np.array([medslope_ecco2-lo_slope_ecco2,up_slope_ecco2-medslope_ecco2])[:,np.newaxis],fmt='o',color='C3')
    #ORAS5
    medslope_ora,medint_ora,lo_slope_ora,up_slope_ora = stats.theilslopes(AMOC['ORAS5'].squeeze(),Sflux['ORAS5'].squeeze())
    ax.errorbar(np.array([np.mean(CO2_values[:40])]),medslope_ora,
                yerr=np.array([medslope_ora-lo_slope_ora,up_slope_ora-medslope_ora])[:,np.newaxis],
                fmt='d',color='C4',label='ORAS5')
    for jj,j in enumerate([5,15,25]):
        medslope_ora1,medint_ora1,lo_slope_ora1,up_slope_ora1 = stats.theilslopes(AMOC['ORAS5'].squeeze()[j:j+15],Sflux['ORAS5'].squeeze()[j:j+15])
        ax.errorbar(np.array([np.mean(CO2_values[j:j+15])]),medslope_ora1,
                yerr=np.array([medslope_ora1-lo_slope_ora1,up_slope_ora1-medslope_ora1])[:,np.newaxis],fmt='o',color='C4')
    #
    ax.set_xlim(0.99,4.01)
    ax.legend()
    ax.set_ylabel(r'AMOC-S$^{34 \degree S}_{transport}$ slope [Sv / ( g/kg $\times$ Sv)]',fontsize=20)
    ax.set_xlabel(r'CO$_2$ forcing',fontsize=20)
    #
    ax2.plot(rho_axis[:3],np.array(s_med_mean)[:3],color='k',marker='o',ls='-')
    ax2.plot(rho_axis[:3],np.array(s_lo)[:3],ls='--',color='k')
    ax2.plot(rho_axis[:3],np.array(s_up)[:3],ls='--',color='k')
    ax2.plot(rho_axis[3:],np.array(s_med_mean)[3:],color='k',marker='o',ls='-')
    ax2.plot(rho_axis[3:],np.array(s_lo)[3:],ls='--',color='k')
    ax2.plot(rho_axis[3:],np.array(s_up)[3:],ls='--',color='k')
    #
    rho_1pct=butils.eosben07_sig0(0,SST_SPG['1pct'],SSS_SPG['1pct']).rolling(year=30,center=True).mean().mean('ens')
    if rho_abs:
        rho_axis_1pct=rho_1pct
    else:
        rho_axis_1pct=-100*(rho_1pct-rho_control)/(rho_control-1000)
    #
    ax2.plot(rho_axis_1pct[15:-15],np.array(s_med_1pct),color='C1',label='1pct')
    ax2.fill_between(rho_axis_1pct[15:-15],np.array(s_lo_1pct),np.array(s_up_1pct),color='C1',alpha=0.25)
    #
    # STABLE SLOPE
    #
    ax2.axhline(y=medslope_stable,color='gray',ls='-')
    ax2.fill_between(np.array([np.nanmin([np.nanmin(rho_axis),np.nanmin(rho_axis_1pct[15:-15])]),
                              np.nanmax([np.nanmax(rho_axis),np.nanmax(rho_axis_1pct[15:-15])])]),
                    lo_slope_stable,up_slope_stable,color='gray',alpha=0.25)
    # ECCO
    if rho_abs:
        sigma0_ecco=butils.eosben07_sig0(0,SST_SPG['ecco'],SSS_SPG['ecco'],pref=0)
        rho_axis_ecco_mean = np.array([np.nanmean(sigma0_ecco.values)])
        rho_axis_ecco1      = np.array([np.nanmean(sigma0_ecco.values[:15])])
        rho_axis_ecco2      = np.array([np.nanmean(sigma0_ecco.values[-15:])])
    else:
        sigma0_ecco=butils.eosben07_sig0(0,SST_SPG['ecco'],SSS_SPG['ecco'],pref=0)-1000
        scale_factor       = -100/sigma0_control
        rho_axis_ecco_mean = scale_factor*np.array([(np.nanmean(sigma0_ecco.values)-sigma0_control)])
        rho_axis_ecco1      = scale_factor*np.array([(np.nanmean(sigma0_ecco.values[:15])-sigma0_control)])
        rho_axis_ecco2      = scale_factor*np.array([(np.nanmean(sigma0_ecco.values[-15:])-sigma0_control)])
    ax2.errorbar(rho_axis_ecco_mean,
                medslope_ecco,yerr=np.array([medslope_ecco-lo_slope_ecco,up_slope_ecco-medslope_ecco])[:,np.newaxis],
                fmt='d',color='C3',label='ECCO')
    ax2.errorbar(rho_axis_ecco1,medslope_ecco1,
                yerr=np.array([medslope_ecco1-lo_slope_ecco1,up_slope_ecco1-medslope_ecco1])[:,np.newaxis],fmt='o',color='C3')
    ax2.errorbar(rho_axis_ecco2,medslope_ecco2,
                yerr=np.array([medslope_ecco2-lo_slope_ecco2,up_slope_ecco2-medslope_ecco2])[:,np.newaxis],fmt='o',color='C3')
    #ORAS5
    if rho_abs:
        sigma0_ora=butils.eosben07_sig0(0,SST_SPG['ORAS5'],SSS_SPG['ORAS5'],pref=0)
        rho_axis_oras5_mean = np.array([np.nanmean(sigma0_ora.values)])
    else:
        sigma0_ora=butils.eosben07_sig0(0,SST_SPG['ORAS5'],SSS_SPG['ORAS5'],pref=0)-1000
        rho_axis_oras5_mean=scale_factor*np.array([(np.mean(sigma0_ora.values)-sigma0_control)])
    ax2.errorbar(rho_axis_oras5_mean,
                medslope_ora,yerr=np.array([medslope_ora-lo_slope_ora,up_slope_ora-medslope_ora])[:,np.newaxis],
                fmt='d',color='C4',label='ORAS5')
    for jj,j in enumerate([5,15,25]):
        medslope_ora1,medint_ora1,lo_slope_ora1,up_slope_ora1 = stats.theilslopes(AMOC['ORAS5'].squeeze()[j:j+15],Sflux['ORAS5'].squeeze()[j:j+15])
        if rho_abs:
            rho_axis_oras5=np.array([np.mean(sigma0_ora[j:j+15])])
        else:
            rho_axis_oras5=-100*np.array([(np.mean(sigma0_ora[j:j+15])-sigma0_control)])/sigma0_control
        ax2.errorbar(rho_axis_oras5,medslope_ora1,
                yerr=np.array([medslope_ora1-lo_slope_ora1,up_slope_ora1-medslope_ora1])[:,np.newaxis],fmt='o',color='C4')
    #
    ax2.set_xlim(np.nanmin([np.nanmin(rho_axis),np.nanmin(rho_axis_1pct[15:-15])]),np.nanmax([np.nanmax(rho_axis),np.nanmax(rho_axis_1pct[15:-15])]))
    extra_artists=[]
    for a,ax1 in enumerate([ax,ax2]):
        txt1=ax1.text(0.0, 1.02, string.ascii_lowercase[a],transform=ax1.transAxes, fontsize=20)
        extra_artists.append(txt1)
    #
    fig.subplots_adjust(wspace=0.05)
    if rho_abs:
        ax2.set_xlim(np.nanmax([np.nanmax(rho_axis),np.nanmax(rho_axis_1pct[15:-15])]),np.nanmin([np.nanmin(rho_axis),np.nanmin(rho_axis_1pct[15:-15])]))
        ax2.set_xlabel(r'SPG potential density [kg m$^{-3}$]',fontsize=20)
        fig.savefig('../Figures/AMOC_Stransport_slope_as_function_of_CO2_and_SPG_density_absolute_new.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    else:
        ax2.set_xlabel('Reduction in SPG potential density anomaly [%]',fontsize=20)
        fig.savefig('../Figures/AMOC_Stransport_slope_as_function_of_CO2_and_SPG_density_new.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    #
    # PLOT DIFFERENT TIMESERIES
    # FIGURES S5-S6
    # 
    fig1,axes1=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    fig2,axes2=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    fig3,axes3=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    fig12,axes12=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    fig22,axes22=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    fig32,axes32=plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,10))
    for c,case in enumerate(names.keys()):
        print(case)
        for e in range(Sflux[case].ens.size):
            sigma0=butils.eosben07_sig0(0,SST_SPG[case].isel(ens=e),SSS_SPG[case].isel(ens=e),pref=0)-1000
            axes12.flatten()[c].set_title(case)
            axes22.flatten()[c].set_title(case)
            axes32.flatten()[c].set_title(case)
            #
            r1,p1=stats.pearsonr(Sflux[case].isel(ens=e)/1E6,sigma0)
            r2,p2=stats.pearsonr(AMOC[case].isel(ens=e)/1E9,sigma0)
            r3,p3=stats.pearsonr(Sflux[case].isel(ens=e)/1E6,AMOC[case].isel(ens=e)/1E9)
            #Scatter
            axes1.flatten()[c].plot(Sflux[case].isel(ens=e)/1E6,sigma0,'.',color='C'+str(c))
            axes1.flatten()[c].set_title(r' '+case+', $r^2$='+str(np.round(r1**2,decimals=2)))
            #
            axes2.flatten()[c].plot(sigma0,AMOC[case].isel(ens=e)/1E9,'.',color='C'+str(c))
            axes2.flatten()[c].set_title(r' '+case+', $r^2$='+str(np.round(r2**2,decimals=2)))
            #
            axes3.flatten()[c].plot(Sflux[case].isel(ens=e)/1E6,AMOC[case].isel(ens=e)/1E9,'.',color='C'+str(c))
            axes3.flatten()[c].set_title(r' '+case+', $r^2$='+str(np.round(r3**2,decimals=2)))
            #timeseries
            axes12.flatten()[c].plot(sigma0.year,sigma0,color='C0',label=r'$\rho_0$')
            axes12_twin=axes12.flatten()[c].twinx()
            axes12_twin.plot(Sflux[case].isel(ens=e).year,Sflux[case].isel(ens=e)/1E6,
                                   color='C1',label=r'$F_S$')
            axes12.flatten()[c].grid(axis='x')
            #
            axes22.flatten()[c].plot(sigma0.year,sigma0,color='C0',label=r'$\rho_0$')
            axes22_twin = axes22.flatten()[c].twinx()
            axes22_twin.plot(AMOC[case].isel(ens=e).year,AMOC[case].isel(ens=e)/1E9,
                             color='C1',label=r'AMOC')
            #
            axes32.flatten()[c].plot(Sflux[case].isel(ens=e).year,Sflux[case].isel(ens=e)/1E6,color='C0',label=r'$F_S$')
            axes32_twin = axes32.flatten()[c].twinx()
            axes32_twin.plot(AMOC[case].isel(ens=e).year,AMOC[case].isel(ens=e)/1E9,
                             color='C1',label=r'AMOC')
            #
            axes12.flatten()[c].grid(axis='x')
            axes12_twin.set_ylim(-25,-60)
            axes22_twin.set_ylim(5,23)
            axes32_twin.set_ylim(5,23)
            if c==0:
                axes12.flatten()[c].legend(loc=2)
                axes22.flatten()[c].legend(loc=2)
                axes32.flatten()[c].legend(loc=2)
                axes22_twin.legend(loc=3)
                axes12_twin.legend(loc=3)
                axes32_twin.legend(loc=3)
    #
    for ax in [axes1.flatten()[-1],axes12.flatten()[-1],axes2.flatten()[-1],axes22.flatten()[-1],axes3.flatten()[-1],axes32.flatten()[-1]]:
        ax.set_title('ECCO and ORAS5')
    sigma0_ecco = butils.eosben07_sig0(0,SST_SPG['ecco'],SSS_SPG['ecco'],pref=0)-1000
    sigma0_ora  = butils.eosben07_sig0(0,SST_SPG['ORAS5'],SSS_SPG['ORAS5'],pref=0)-1000
    #sigma0_ecco=gsw.sigma0(SSS_SPG['ecco'], gsw.CT_from_pt(SSS_SPG['ecco'], SST_SPG['ecco']))
    #sigma0_ora=gsw.sigma0(SSS_SPG['ORAS5'], gsw.CT_from_pt(SSS_SPG['ORAS5'], SST_SPG['ORAS5']))
    axes1.flatten()[-1].plot(Sflux['ecco'].isel(ens=e),sigma0_ecco,'.',color='k',label='ECCO')
    axes1.flatten()[-1].plot(Sflux['ORAS5'].isel(ens=e),sigma0_ora,'d',color='r',label='ORAS5')
    axes2.flatten()[-1].plot(sigma0_ecco,AMOC['ecco'].isel(ens=e),'.',color='k',label='ECCO')
    axes2.flatten()[-1].plot(sigma0_ora,AMOC['ORAS5'].isel(ens=e),'d',color='r',label='ORAS5')
    axes3.flatten()[-1].plot(Sflux['ecco'].isel(ens=e),AMOC['ecco'].isel(ens=e),'.',color='k',label='ECCO')
    axes3.flatten()[-1].plot(Sflux['ORAS5'].isel(ens=e),AMOC['ORAS5'].isel(ens=e),'d',color='r',label='ORAS5')
    for ax in [axes12.flatten()[-1],axes22.flatten()[-1]]:
        ax.plot(sigma0_ecco.time.groupby('time.year').mean().year-1979,sigma0_ecco,color='C0',label=r'$\rho^{ECCO}_0$')
        ax.plot(sigma0_ora.time_counter.groupby('time_counter.year').mean().year-1979,sigma0_ora,color='C0',ls='--',label=r'$\rho^{ORAS5}_0$')
    #
    axes32.flatten()[-1].plot(sigma0_ecco.time.groupby('time.year').mean().year-1979,Sflux['ecco'].isel(ens=e),color='C1',label=r'$F^{ECCO}_S$')
    axes32.flatten()[-1].plot(sigma0_ora.time_counter.groupby('time_counter.year').mean().year-1979,Sflux['ORAS5'].isel(ens=e),color='C1',ls='--',label=r'$F^{ORAS5}_S$')
    #
    axes12_twin = axes12.flatten()[-1].twinx()
    axes22_twin = axes22.flatten()[-1].twinx()
    axes32_twin = axes32.flatten()[-1].twinx()
    axes12_twin.plot(sigma0_ecco.time.groupby('time.year').mean().year-1979,Sflux['ecco'].isel(ens=e),color='C1',label=r'$F^{ECCO}_S$')
    axes12_twin.plot(sigma0_ora.time_counter.groupby('time_counter.year').mean().year-1979,Sflux['ORAS5'].isel(ens=e),color='C1',ls='--',label=r'$F^{ORAS5}_S$')
    axes22_twin.plot(sigma0_ecco.time.groupby('time.year').mean().year-1979,AMOC['ecco'].isel(ens=e),color='C1',label=r'$AMOC^{ECCO}_S$')
    axes22_twin.plot(sigma0_ora.time_counter.groupby('time_counter.year').mean().year-1979,AMOC['ORAS5'].isel(ens=e),color='C1',ls='--',label=r'$AMOC^{ORAS5}_S$')
    axes32_twin.plot(sigma0_ecco.time.groupby('time.year').mean().year-1979,AMOC['ecco'].isel(ens=e),color='C1',label=r'$AMOC^{ECCO}_S$')
    axes32_twin.plot(sigma0_ora.time_counter.groupby('time_counter.year').mean().year-1979,AMOC['ORAS5'].isel(ens=e),color='C1',ls='--',label=r'$AMOC^{ORAS5}_S$')
    axes12_twin.set_ylim(-25,-60)
    axes22_twin.set_ylim(5,23)
    axes32_twin.set_ylim(5,23)
    #
    axes32.flatten()[-1].set_ylim(-25,-60)
    axes12.flatten()[-1].legend(loc=1)
    axes22.flatten()[-1].legend(loc=1)
    axes32.flatten()[-1].legend(loc=1)
    axes12_twin.legend(loc=4)
    axes22_twin.legend(loc=4)
    axes32_twin.legend(loc=4)
    #
    xlab=fig1.text(0.5,0.05,r'Salt Transport [g/kg $\cdot$ Sv]',fontsize=20,ha='center',va='center')
    ylab=fig1.text(0.05,0.5,r'$\sigma_0$ [g/kg]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig1.savefig('../Figures/rho_Sflux_scatter.png',dpi=300,bbox_extra_artists=[xlab,ylab])
    xlab=fig2.text(0.5,0.05,r'$\sigma_0$ [g/kg]',fontsize=20,ha='center',va='center')
    ylab=fig2.text(0.05,0.5,r'AMOC @26N [Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig2.savefig('../Figures/rho_AMOC_scatter.png',dpi=300,bbox_extra_artists=[xlab,ylab])
    xlab=fig3.text(0.5,0.05,r'Salt Transport [g/kg $\cdot$ Sv]',fontsize=20,ha='center',va='center')
    ylab=fig3.text(0.05,0.5,r'AMOC @26N [Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig3.savefig('../Figures/Stransport_AMOC_scatter.png',dpi=300,bbox_extra_artists=[xlab,ylab])
    ylab1=fig12.text(0.05,0.5,r'$\sigma_0$ [g/kg]',fontsize=20,rotation='vertical',ha='center',va='center')
    ylab2=fig12.text(0.96,0.5,r'Salt Transport [g/kg $\cdot$ Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig12.savefig('../Figures/rho_Sflux_timeseries.png',dpi=300,bbox_extra_artists=[ylab1,ylab2])
    ylab1=fig22.text(0.05,0.5,r'$\sigma_0$ [g/kg]',fontsize=20,rotation='vertical',ha='center',va='center')
    ylab2=fig22.text(0.96,0.5,r'AMOC @26N [Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig22.savefig('../Figures/rho_AMOC_timeseries.png',dpi=300,bbox_extra_artists=[ylab1,ylab2])
    ylab1=fig32.text(0.05,0.5,r'Salt Transport [g/kg $\cdot$ Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    ylab2=fig32.text(0.96,0.5,r'AMOC @26N [Sv]',fontsize=20,rotation='vertical',ha='center',va='center')
    fig32.savefig('../Figures/Stransport_AMOC_timeseries.png',dpi=300,bbox_extra_artists=[ylab1,ylab2])
    #
    #
    #################################
    # FIGURE S8: INTERANNUAL VARIABILITY VERSUS TREND
    #
    trends={}
    stds={}
    dy=30
    for c,case in enumerate(names.keys()):
        print(case)
        s_med_e=[]
        s_up_e=[]
        s_lo_e=[]
        s_std_e=[]
        for e in range(Sflux[case].ens.size):
            sigma0=butils.eosben07_sig0(0,SST_SPG[case].isel(ens=e),SSS_SPG[case].isel(ens=e),pref=0)-1000
            var_in = -(Sflux_1000_all[case].sel(lat=60).isel(ens=e) \
                                  -Sflux_1000_all[case].sel(lat=40).isel(ens=e))/1E6
            s_med=[]
            s_medint=[]
            s_up=[]
            s_lo=[]
            s_std=[]
            for n in range(0,Sflux[case].year.size-dy,1):
                medslope_1,medint_1,lo_slope_1,up_slope_1=stats.theilslopes(AMOC[case].isel(ens=e).squeeze().values[n:n+dy]/1E9)
                medslope_2,medint_2,lo_slope_2,up_slope_2=stats.theilslopes(sigma0.squeeze().values[n:n+dy])
                medslope_3,medint_3,lo_slope_3,up_slope_3=stats.theilslopes(var_in.squeeze().values[n:n+dy]/1E6)
                s_med.append([medslope_1,medslope_2,medslope_3])
                s_lo.append([lo_slope_1,lo_slope_2,lo_slope_3])
                s_up.append([up_slope_1,up_slope_2,up_slope_3])
                s_medint.append([medint_1,medint_2,medint_3])
                s_std.append([np.std(AMOC[case].isel(ens=e).squeeze().values[n:n+dy]/1E9-(np.arange(dy)*medslope_1+medint_1)),
                              np.std(sigma0.squeeze().values[n:n+dy]-(np.arange(dy)*medslope_2+medint_2)),
                              np.std(var_in.squeeze().values[n:n+dy]/1E6-(np.arange(dy)*medslope_3+medint_3))])
            s_med_e.append(np.array(s_med))
            s_up_e.append(np.array(s_up))
            s_lo_e.append(np.array(s_lo))
            s_std_e.append(np.array(s_std))
        trends[case]=np.nanmean(np.array(s_med_e),0)
        trends[case+'_up']=np.nanmean(np.array(s_up_e),0)
        trends[case+'_lo']=np.nanmean(np.array(s_lo_e),0)
        stds[case]=np.nanmean(np.array(s_std_e),0)
    #
    fig,axes = plt.subplots(nrows=3,ncols=2,sharex=False,figsize=(15,10))
    # 
    axes[0,0].set_title('AMOC SNR')
    axes[1,0].set_title(r'SPG $\sigma_0$ SNR')
    axes[2,0].set_title(r'Salt transport convergence SNR')
    std_mean=np.mean(stds['x1.0'],axis=0)
    SNR0=abs(trends['x1.0'])/stds['x1.0']
    #print(std_mean)
    dy2=30
    SNR_all=[]
    for c,case in enumerate(names.keys()):
        timeaxis=np.arange(dy/2,Sflux[case].year.size-dy/2,1)
        SNR = abs(trends[case])/stds[case]
        SNR_all.append(np.max(SNR,axis=0))
        STD = stds[case]/std_mean
        is_significant=[]
        for n in range(0,Sflux[case].year.size-dy-dy2,1):
            results=stats.ttest_ind(SNR[n:n+dy2,:],SNR0,axis=0,equal_var=False,alternative='greater')
            is_significant.append(results.pvalue<0.01)
        is_significant=np.array(is_significant)
        for j in range(3):
            axes[j,0].plot(timeaxis,SNR[:,j],label=case)
            sig_inds=np.where(is_significant[:,j])[0]
    #
    axes[0,0].legend(ncols=4)
    for ax in axes.flatten():
        ax.set_ylim(0,0.7)
    for a,ax in enumerate(axes[:,-1].flatten()):
        ax.plot(np.array([1.0,1.4,1.6,1.8,2,2.8,4]),np.array(SNR_all)[:-1,a],ls='-',marker='.',color='k')
        ax.axhline(y=np.array(SNR_all)[-1,a],ls='--',lw=2,color='gray',label='1pct')
        ax.yaxis.tick_right()
    #
    ylab1=fig.text(0.05,0.5,r'Signal to noise ratio',fontsize=20,rotation='vertical',ha='center',va='center')
    ylabel=fig.text(0.97,0.5,r'Maximum signal to noise ratio',fontsize=20,rotation='vertical',ha='center',va='center')
    xlab=axes[-1,0].set_xlabel(r'Time [years]',fontsize=20)
    xlab2=axes[-1,1].set_xlabel(r'CO$_2$ forcing',fontsize=20)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig('../Figures/Signal_to_noise_ratio_running_'+str(dy)+'_years.png',dpi=300, bbox_extra_artists=[ylab1,xlab])
        
    ###########################
    # FIGURE S7: HOVMOELLER PLOT (DEPTH-TIME) OF
    # OVERTURNING STREAMFUNCTION AND SALINITY AT 34S
    if False:
        from scipy import interpolate
        case='x4.0'
        for n,name in enumerate(names[case]['name'][:3]):
            print(n,name)
            filelist=sorted(glob.glob(names[case]['path'][n]+name+'/ocn/hist/'+name+'.*.hm.*-*.nc'))[:100*12]
            print(len(filelist))
            data=xr.open_mfdataset(filelist,combine='nested',concat_dim='time',chunks={'time':120})
            if n==0:
                AMOC_z=(data['mmflxd'].isel(region=0).sel(lat=-34).groupby('time.year').mean()/1E9).expand_dims('ens').compute()
                S_zmean=((data.salnlvl.isel(y=jinds_34S,x=iinds_34S)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points') \
                    /(data.salnlvl.isel(y=jinds_34S,x=iinds_34S).notnull().astype(float)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points')). \
                    groupby('time.year').mean().expand_dims('ens').compute()
                AMOC_z = AMOC_z.assign_coords({'year':range(AMOC_z.year.size)})
                S_zmean=S_zmean.assign_coords({'year':range(S_zmean.year.size)})
            else:
                AMOC_z_dum = (data['mmflxd'].isel(region=0).sel(lat=-34).groupby('time.year').mean()/1E9).expand_dims('ens').compute()
                S_zmean_dum=((data.salnlvl.isel(y=jinds_34S,x=iinds_34S)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points') \
                    /(data.salnlvl.isel(y=jinds_34S,x=iinds_34S).notnull().astype(float)*grid_blom.parea.isel(y=jinds_34S,x=iinds_34S)).sum('points')). \
                    groupby('time.year').mean().expand_dims('ens').compute()
                AMOC_z_dum = AMOC_z_dum.assign_coords({'year':range(AMOC_z_dum.year.size)})
                S_zmean_dum=S_zmean_dum.assign_coords({'year':range(S_zmean_dum.year.size)})
                AMOC_z  = xr.concat([AMOC_z_dum,AMOC_z],dim='ens')
                S_zmean = xr.concat([S_zmean_dum,S_zmean],dim='ens') 
        #
        AMOC_1depth=[]
        AMOC_5depth=[]
        AMOC_m1depth=[]
        AMOC_maxdepth=[]
        for year in AMOC_z.year:
            dum=interpolate.interp1d(AMOC_z.mean('ens').isel(year=year.values)[20:60],AMOC_z.depth[20:60].values+AMOC_z.depth.diff(dim='depth')[20:60].values)
            AMOC_maxdepth.append(dum(AMOC_z.mean('ens').isel(year=year.values).max()))
            AMOC_1depth.append(dum(1.0))
            AMOC_m1depth.append(dum(-.5))
            AMOC_5depth.append(dum(5.0))
        #
        AMOC_maxdepth = xr.DataArray(np.array(AMOC_maxdepth),coords={'year':AMOC_z.year})
        AMOC_1depth   = xr.DataArray(np.array(AMOC_1depth),coords={'year':AMOC_z.year})
        AMOC_m1depth  = xr.DataArray(np.array(AMOC_m1depth),coords={'year':AMOC_z.year})
        AMOC_5depth   = xr.DataArray(np.array(AMOC_5depth),coords={'year':AMOC_z.year})
        #
        xr.merge([AMOC_z.to_dataset(name='AMOC_z'),
                  S_zmean.to_dataset(name='S_zmean'),
                  AMOC_maxdepth.to_dataset(name='AMOC_maxdepth'),
                  AMOC_1depth.to_dataset(name='AMOC_1depth'),
                  AMOC_m1depth.to_dataset(name='AMOC_m1depth'),
                  AMOC_5depth.to_dataset(name='AMOC_5depth')]).to_netcdf('../data/NorESM2-LM/AMOC_S_t-z_at34S.nc')
    #
    AMOC_S_tz = xr.open_dataset('../data/NorESM2-LM/AMOC_S_t-z_at34S.nc')
    #
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,5),sharex=True,sharey=True)
    AMOC_S_tz.AMOC_z.mean('ens').T.plot(ax=ax,
                              cbar_kwargs={'label':r'AMOC@34$\degree$S [Sv]'})
    cs1=AMOC_S_tz.S_zmean.mean('ens').T.plot.contour(ax=ax,levels=[34.4,34.6,34.8,34.9,35],colors='k')
    AMOC_S_tz.AMOC_1depth.plot(ax=ax,color='C0',lw=2,ls='--')
    AMOC_S_tz.AMOC_m1depth.plot(ax=ax,color='C0',lw=2,ls=':')
    AMOC_S_tz.AMOC_5depth.plot(ax=ax,color='C0',lw=2,ls='-.')
    AMOC_S_tz.AMOC_maxdepth.plot(ax=ax,color='C0',lw=2,ls='-')
    ax.set_ylim(5000,0)
    ax.set_ylabel('Depth [m]', fontsize=18)
    ax.set_xlabel('Time [year]',fontsize=18)
    ax.set_title('')
    ax.clabel(cs1)
    fig.savefig('../Figures/AMOC_Salt_depth_time_at34S_v2.png',dpi=300)
    #
    ######################################################
    #
    # FIGURE 3: 0-CROSSING TIME AND LEAD LAG ANALYSIS
    #
    AMOC_0crosstime=[]
    Sflux_0crosstime=[]
    for c,case in enumerate(list(names.keys())[:-1]):
            print(case)
            s_AMOC=[]
            s_Sflux=[]
            for e in range(Sflux[case].ens.size):
                ll = np.where(np.isfinite(Sflux[case].isel(ens=e)))[0]
                dum_AMOC=[]
                dum_Sflux=[]
                Troll=10
                years=np.arange(Troll/2,AMOC[case].isel(ens=e).size-30-Troll/2,1).astype(int)
                for j in years:
                    results_AMOC  = stats.theilslopes(AMOC[case].isel(ens=e).values[j:j+30]/1E9)
                    results_Sflux = stats.theilslopes(Sflux[case].isel(ens=e).values[j:j+30]/1E6)
                    dum_AMOC.append(results_AMOC.slope)
                    dum_Sflux.append(results_Sflux.slope)
                AMOC_0cross=np.where(np.diff(np.signbit(np.array(dum_AMOC))))[0]
                Sflux_0cross=np.where(np.diff(np.signbit(np.array(dum_Sflux))))[0]
                if len(AMOC_0cross)>0:
                    s_AMOC.append(AMOC_0cross[0]+15+Troll/2) #center the trend
                else:
                    s_AMOC.append(np.nan)
                if len(Sflux_0cross)>0:
                    s_Sflux.append(Sflux_0cross[0]+15+Troll/2) #center the trend
                else:
                    s_Sflux.append(np.nan)
            AMOC_0crosstime.append(s_AMOC)
            Sflux_0crosstime.append(s_Sflux)
    #
    AMOCmean_0crossing=[]
    Sfluxmean_0crossing=[]
    co2_axis=np.array([1.0,1.4,1.6,1.8,2,2.8,4])
    fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(18,7.5),sharex=False,sharey=False)
    for s in range(len(AMOC_0crosstime)):
        AMOCmean_0crossing.append(np.mean(AMOC_0crosstime[s]))
        Sfluxmean_0crossing.append(np.mean(Sflux_0crosstime[s]))
        if len(AMOC_0crosstime[s])>1:
            ax1.errorbar(co2_axis[s],np.mean(AMOC_0crosstime[s]),yerr=np.array([np.mean(AMOC_0crosstime[s])-np.min(AMOC_0crosstime[s]),
                                                                               np.max(AMOC_0crosstime[s])-np.mean(AMOC_0crosstime[s])])[:,np.newaxis],color='C0')
            ax1.errorbar(co2_axis[s],np.mean(Sflux_0crosstime[s]),yerr=np.array([np.mean(Sflux_0crosstime[s])-np.min(Sflux_0crosstime[s]),
                                                                                np.max(Sflux_0crosstime[s])-np.mean(Sflux_0crosstime[s])])[:,np.newaxis],color='C1')
    ax1.plot(co2_axis,np.array(AMOCmean_0crossing),ls='-',marker='o',color='C0', label=r'AMOC @ 26$\degree$N')
    ax1.plot(co2_axis,np.array(Sfluxmean_0crossing),ls='-',marker='o',color='C1', label=r'S$_{transport}$ @ 34$\degree$S')
    ax1.legend(fontsize=18)
    ax1.set_xlim(0.99,4.01)
    ax1.set_ylabel('Trend 0-crossing time [year]',fontsize=20)
    ax1.set_xlabel('CO2 forcing',fontsize=20)
    #fig.savefig('AMOC_Sflux_0crossing.png',dpi=300)
    # Sflux - AMOC lagged correlation                                                                                                                                                                                                                
    for c,case in enumerate(list(names.keys())[:-1]):
        ax2.plot(lags,np.array(r_all[case]).squeeze()**2,label=case)
    ax2.legend(fontsize=18)
    ax2.axvline(x=0,color='gray',lw=2,ls='--')
    ax2.set_ylabel(r'Pearson r$^2$',fontsize=20)
    ax2.set_xlabel(r'Lag [years]',fontsize=20)
    ax2.set_xlim(np.min(lags),np.max(lags))
    extra_artists=[]
    for a,ax in enumerate([ax1,ax2]):
        txt1=ax.text(0.0, 1.02, string.ascii_lowercase[a],transform=ax.transAxes, fontsize=20)
        extra_artists.append(txt1)
    fig.savefig('../Figures/AMOC_Sflux_0crossing_and_LAGGED_correlation_annual.png',dpi=300,bbox_inches='tight',bbox_extra_artists=extra_artists)
    #print out the mean lag
    for c,case in enumerate(list(names.keys())):
        print('Mean Lag',case,str(np.sum(lags*abs(np.array(r_all[case]).squeeze()))/np.sum(abs(np.array(r_all[case])).squeeze())))
