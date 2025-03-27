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
    for year in [sys.argv[1]]:
        print(year)
        s_paths=[]
        t_paths=[]
        vol_paths=[]
        for month in range(1,13):
            print(month)
            s_paths.append('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/vosaline/opa0/vosaline_ORAS5_1m_'+str(year)+str(month).zfill(2)+'_grid_T_02.nc')
            t_paths.append('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/votemper/opa0/votemper_ORAS5_1m_'+str(year)+str(month).zfill(2)+'_grid_T_02.nc')
            vol_paths.append('https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/EASYInit/oras5/ORCA025/vomecrtn/opa0/vomecrtn_ORAS5_1m_'+str(year)+str(month).zfill(2)+'_grid_T_02.nc')
        #
        sal=xr.open_mfdataset(s_paths)
        vol=xr.open_mfdataset(vol_paths)
        #
        #
        if upper_1000:
                AMOC=(vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['26N']['x'],y=coords['26N']['y']) \
                    .sum('x').fillna(0).cumulative_integrate('deptht').max('deptht').compute()
                Sflux=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['34S']['x'],y=coords['34S']['y']) \
                   .sum('x').fillna(0).isel(deptht=zslice).integrate('deptht').compute()
                Sflux_26N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['26N']['x'],y=coords['26N']['y']) \
                   .sum('x').fillna(0).isel(deptht=zslice).integrate('deptht').compute()
                Sflux_29N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['29N']['x'],y=coords['29N']['y']) \
                   .sum('x').fillna(0).isel(deptht=zslice).integrate('deptht').compute()
                Sflux_40N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['40N']['x'],y=coords['40N']['y']) \
                   .sum('x').fillna(0).isel(deptht=zslice).integrate('deptht').compute()
                Sflux_60N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['60N']['x'],y=coords['60N']['y']) \
                   .sum('x').fillna(0).isel(deptht=zslice).integrate('deptht').compute()
                file_extent='_new_1000.nc'
        else:
                AMOC=(vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['26N']['x'],y=coords['26N']['y']) \
                    .sum('x').fillna(0).cumulative_integrate('deptht').max('deptht').compute()
                Sflux=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['34S']['x'],y=coords['34S']['y']) \
                    .sum('x').fillna(0).integrate('deptht').compute()
                Sflux_26N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['26N']['x'],y=coords['26N']['y']) \
                   .sum('x').fillna(0).integrate('deptht').compute()
                Sflux_29N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['29N']['x'],y=coords['29N']['y']) \
                   .sum('x').fillna(0).integrate('deptht').compute()
                Sflux_40N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['40N']['x'],y=coords['40N']['y']) \
                   .sum('x').fillna(0).integrate('deptht').compute()
                Sflux_60N=(sal.vosaline*vol.vomecrtn*mesh.e1v.squeeze()).isel(x=coords['60N']['x'],y=coords['60N']['y']) \
                   .sum('x').fillna(0).integrate('deptht').compute()
                file_extent='.nc'
                #
        xr.merge([AMOC.to_dataset(name='AMOC'),
                  Sflux.to_dataset(name='Sflux'),
                  Sflux_26N.to_dataset(name='Sflux_26N'),
                  Sflux_29N.to_dataset(name='Sflux_29N'),
                  Sflux_40N.to_dataset(name='Sflux_40N'),
                  Sflux_60N.to_dataset(name='Sflux_60N'),]). \
                  to_netcdf('/projects/NS9874K/anu074/ORAS5/AMOC_Sflux_'+str(year)+str(month).zfill(2)+file_extent)
        vol.close()
        sal.close()
    #
    if False:
        ORAS5 = xr.open_mfdataset(sorted(glob.glob('/projects/NS9874K/anu074/ORAS5/AMOC_Sflux*_new.nc')))
        ORAS5 = ORAS5.rename({'time_counter':'time'})
        ORAS5 = ORAS5.groupby('time.year').mean().compute()
        #
        ORAS5_1000 = xr.open_mfdataset(sorted(glob.glob('/projects/NS9874K/anu074/ORAS5/AMOC_Sflux*_new_1000.nc')))
        ORAS5_1000 = ORAS5_1000.rename({'time_counter':'time'})
        ORAS5_1000 = ORAS5_1000.groupby('time.year').mean().compute()
        ORAS5_1000 = ORAS5_1000.expand_dims({'ens':np.array([0])})
        # 
        ORAS5_1000.to_netcdf('/projects/NS9874K/anu074/ORAS5/AMOC_Sflux_0_1000_1979_2018.nc')
        ORAS5.to_netcdf('/projects/NS9874K/anu074/ORAS5/AMOC_Sflux_1979_2018.nc')
