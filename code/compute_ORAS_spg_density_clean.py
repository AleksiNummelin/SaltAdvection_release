import xarray as xr
import numpy as np
import gsw
import dask, distributed
import os
import socket
import sys

if __name__ == '__main__':
    local_dir = '/scratch/anu074/'+socket.gethostname()+'/'
    if not os.path.isdir(local_dir):
        os.system('mkdir -p '+local_dir)
        print('created folder '+local_dir)
    #
    n_workers = 2
    n_threads = 2
    processes = True
    cluster   = distributed.LocalCluster(n_workers=n_workers,threads_per_worker=n_threads,processes=processes,
                                              local_directory=local_dir,lifetime='48 hour',lifetime_stagger='10 minutes',
                                              lifetime_restart=True,dashboard_address=None,worker_dashboard_address=None)
    client    = distributed.Client(cluster)
    #
    # these follow hard coded coordinates - closes to a latitude line, but not exact in the north due to tripolar grid
    #
    coords={'34S':{'y':350,'x':slice(-511,-215)},
            '26N':{'y':606,'x':slice(-620,-345)},
            '29N':{'y':620,'x':slice(-620,-330)},
            '40N':{'y':673,'x':slice(852,1114)},
            '60N':{'y':840,'x':slice(870,1170)},
    }
    bathy=xr.open_dataset('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/mesh/bathy_meter.nc')
    mesh=xr.open_dataset('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/mesh/mesh_mask.nc')
    mask=np.logical_and(np.logical_and(np.logical_and(bathy.nav_lat>50,
                                                      bathy.nav_lat<60),
                                           np.logical_and(bathy.nav_lon>-50,
                                                          bathy.nav_lon<-40)),
                            bathy.Bathymetry>2000)
    dA          = (mesh.e1t*mesh.e2t).squeeze()
    zslice      = slice(0,46) #0-1000 m
    jinds,iinds = np.where(mask)
    jinds       = xr.DataArray(jinds,dims='points')
    iinds       = xr.DataArray(iinds,dims='points')
    #
    for year in range(1979,2019):
        print(year)
        s_paths=[]
        t_paths=[]
        vol_paths=[]
        for month in range(1,4):
            print(month)
            s_paths.append('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/vosaline/opa0/vosaline_ORAS5_1m_'+str(year)+str(month).zfill(2)+'_grid_T_02.nc')
            t_paths.append('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/votemper/opa0/votemper_ORAS5_1m_'+str(year)+str(month).zfill(2)+'_grid_T_02.nc')
        #
        sal  = xr.open_mfdataset(s_paths)
        temp = xr.open_mfdataset(t_paths)
        if year==1979:
                SST_SPG = ((dA*temp.votemper).isel(y=jinds,x=iinds).sum('points')/ \
                           dA.isel(y=jinds,x=iinds).sum('points')).resample(time_counter='QS').mean(dim='time_counter')\
                           .isel(time_counter=slice(0,-1,4)).compute()
                SSS_SPG = ((dA*sal.vosaline).isel(y=jinds,x=iinds).sum('points')/ \
                           dA.isel(y=jinds,x=iinds).sum('points')).resample(time_counter='QS').mean(dim='time_counter')\
                           .isel(time_counter=slice(0,-1,4)).compute()
        else:
                SST_dum = ((dA*temp.votemper).isel(y=jinds,x=iinds).sum('points')/ \
                           dA.isel(y=jinds,x=iinds).sum('points')).resample(time_counter='QS').mean(dim='time_counter')\
                           .isel(time_counter=slice(0,-1,4)).compute()
                SSS_dum = ((dA*sal.vosaline).isel(y=jinds,x=iinds).sum('points')/ \
                           dA.isel(y=jinds,x=iinds).sum('points')).resample(time_counter='QS').mean(dim='time_counter')\
                           .isel(time_counter=slice(0,-1,4)).compute()
                SST_SPG = xr.concat([SST_SPG,SST_dum],dim='time_counter')
                SSS_SPG = xr.concat([SSS_SPG,SSS_dum],dim='time_counter')
            #
    xr.merge([SST_SPG.to_dataset(name='SST_SPG'),
              SSS_SPG.to_dataset(name='SSS_SPG')]). \
              to_netcdf('/projects/NS9874K/anu074/ORAS5/SPG_SSS_SST_1979_2018_JFM.nc')
