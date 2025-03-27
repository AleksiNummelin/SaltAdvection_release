import xarray as xr
#import xesmf as xe
#from scipy import interpolate
import numpy as np

def read_sections(filename):
    '''
    Read the section indeces defined on NorESM grid. 
    Note that these indeces are defined for Fortran 1 based system 
    so one needs to substract one to get the correct python 0 based indeces
    '''
    #
    data=open(filename).readlines()
    titles = [name for name in data if 1+name.find('Name')]
    indices = [data.index(title) for title in titles] #indices of the titles
    indices.append(len(data)) # last index
    #
    output={}
    for ii,ind in enumerate(indices[:-1]):
        dum=[]
        for jj in range(ind+1,indices[ii+1]):
            dum.append(np.array(data[jj].strip().split(),dtype=int)-1)
        output[titles[ii][6:-1].strip()]=np.array(dum)
        
    return output

def read_mertra(filename):
    '''
    Read meridional transport indices defined on NorESM grid. 
    Note that these indeces are defined for Fortran 1 based system 
    so one needs to substract one to get the correct python 0 based indeces
    
    data is ordered such that output[lat]=[x_ind, y_ind, x_weight, y_weight]
    '''
    #
    data=open(filename).readlines()
    titles = [name for name in data if 1+name.find('Section')]
    indices = [data.index(title) for title in titles] #indices of the titles
    indices.append(len(data)) # last index
    #
    output={}
    for ii,ind in enumerate(indices[:-1]):
        dum=[]
        for jj in range(ind+1,indices[ii+1]):
            dum_jj     = np.array(data[jj].strip().split(),dtype=int)
            dum_jj[:2] = dum_jj[:2] -1
            dum.append(dum_jj)
        #
        output[float(titles[ii][7:15].strip())]=np.array(dum)
        
    return output

def bottom_vel(v,dz):
    bi=np.where(dz>1)[0]
    if len(bi>0):
        return v[bi[-1]]
    else:
        return np.nan

def NorESM_to_WOA_regridder(NorESM_grid_path='', WOA_grid_path='', grid_weight_path = '',regrid_mode='conservative',reuse_weights=True):
    '''
    
    '''
    # open WOA data to get the grid
    WOA =xr.open_dataset(WOA_grid_path, decode_times=False)
    mask_WOA = WOA.t_an.isel(depth=0,time=0).notnull().drop('depth').drop('time')
    mask_WOA = mask_WOA.rename('mask_WOA').to_dataset().rename_dims({'lon':'x','lat':'y'}).mask_WOA
    # setup WOA grid
    #
    lat = WOA.lat.rename({'lat':'y'})
    lon = WOA.lon.rename({'lon':'x'})
    lat_b = xr.concat([WOA.lat_bnds.isel(nbounds=0),WOA.lat_bnds.isel(lat=-1).isel(nbounds=1)],dim='lat').rename('lat_b').rename({'lat':'y_b'})
    lon_b = xr.concat([WOA.lon_bnds.isel(nbounds=0),WOA.lon_bnds.isel(lon=-1).isel(nbounds=1)],dim='lon').rename('lon_b').rename({'lon':'x_b'})
    #
    WOA_grid = xr.merge([lon,lat,lon_b,lat_b])
    #
    # NorESM grid
    grid = xr.open_dataset(NorESM_grid_path)
    mask_NorESM = grid.pmask.astype(float).rename('mask_NorESM')
    #
    ny,nx = mask_NorESM.shape
    lat_noresm = grid.plat.rename('lat').isel(y=slice(0,ny-1))
    lon_noresm = grid.plon.rename('lon').isel(y=slice(0,ny-1))
    #
    lon_b_noresm = xr.concat([grid.pclon.isel(nv=0),grid.pclon.isel(nv=1,x=-1)],dim='x')
    lon_b_noresm = xr.concat([lon_b_noresm,xr.concat([grid.pclon.isel(nv=3,y=-1),grid.pclon.isel(nv=2,y=-1,x=-1)],dim='x')],dim='y')
    lat_b_noresm= xr.concat([grid.pclat.isel(nv=0),grid.pclat.isel(nv=1,x=-1)],dim='x')
    lat_b_noresm = xr.concat([lat_b_noresm,xr.concat([grid.pclat.isel(nv=3,y=-1),grid.pclat.isel(nv=2,y=-1,x=-1)],dim='x')],dim='y')
    #
    lon_b_noresm = lon_b_noresm.isel(y=slice(0,ny)).rename('lon_b').rename({'y':'y_b','x':'x_b'})
    lat_b_noresm = lat_b_noresm.isel(y=slice(0,ny)).rename('lat_b').rename({'y':'y_b','x':'x_b'})
    #
    NorESM_grid = xr.merge([lon_noresm,lat_noresm,lon_b_noresm,lat_b_noresm])
    #
    # create the regridder
    regridder = xe.Regridder(NorESM_grid, WOA_grid, regrid_mode,filename=grid_weight_path+'NORESM_to_WOA_'+regrid_mode+'.nc',reuse_weights=reuse_weights)
    #
    return regridder, mask_WOA, mask_NorESM

def noresm_bounds(clon,clat):
    '''Return lon,lat bounds based on corner information clon,clat'''
    nv, ny,nx = clon.shape
    lon_b = np.concatenate([clon.values[0,:,:],clon.values[1,:,-1:]],axis=1)
    lon_b = np.concatenate([lon_b,np.concatenate([clon.values[3,-1:,:],clon.values[2,-1:,-1:]],axis=1)],axis=0)
    lat_b = np.concatenate([clat.values[0,:,:],clat.values[1,:,-1:]],axis=1)
    lat_b = np.concatenate([lat_b,np.concatenate([clat.values[3,-1:,:],clat.values[2,-1:,-1:]],axis=1)],axis=0)
    lon_b[np.where(lon_b>180)] = lon_b[np.where(lon_b>180)]-360
    lon_b[np.where(lon_b<-180)] = lon_b[np.where(lon_b<-180)]+360
    #
    lon_b = xr.DataArray(lon_b,dims=('y','x'),name='lon_b')
    lat_b = xr.DataArray(lat_b,dims=('y','x'),name='lat_b')
    #
    return lon_b, lat_b

def noresm2scalar(u,v):
    """Interpolate the velocity components to scalar points on NorESM C grid """
    #
    nx = len(u.x)
    ny = len(u.y)
    # Get velocity components at scalar point
    us=xr.concat([u.isel(x=slice(0,nx-1)),u.isel(x=slice(1,nx))],dim='dum').mean(dim='dum')
    us=xr.concat([us,xr.concat([u.isel(x=0),u.isel(x=-1)],dim='dum').mean(dim='dum')],dim='x')
    vs=xr.concat([v.isel(y=slice(0,ny-1)),v.isel(y=slice(1,ny))],dim='dum').mean(dim='dum')
    vs=xr.concat([vs,v.isel(y=-1)],dim='y')
    #
    return us,vs


def dz2zaxis(dz):
    '''
    Create a zaxis suitable for plotting
    '''
    xlen = len(dz.x)
    ylen = len(dz.y)
    tlen = len(dz.time)
    #
    zaxis = xr.concat([xr.zeros_like(dz.isel(sigma=0)),dz],dim='sigma').cumsum(dim='sigma')
    zaxis = xr.concat([zaxis.isel(x=0),zaxis],dim='x')
    zaxis = xr.concat([zaxis.isel(y=0),zaxis],dim='y')
    #
    return zaxis

def interp_sigma2depth(var, coord_in, coord_out):
    """
    interpolate from coord_in to coord_out
    :var vector: input data in coord_in space
    :coord_in vector: coordinate of the input data
    :coord_out vector: output cooridinate
    """
    #coord_out = np.array(list(range(0,100,10))+list(range(100,1000,100))+list(range(1000,2500,250)))
    #inds = np.where(np.diff(coord_in)>0)[0]
    inds  = np.where(np.isfinite(var))[0]
    if len(inds)>1:
        return np.interp(coord_out, coord_in[inds], var[inds], left=np.nan, right=np.nan)
        #inds2 = np.where(coord_out<=coord_in[-1])[0] #mask values below bottom
        #return np.concatenate([np.interp(coord_out, coord_in[inds], var[inds])[inds2], np.nan*np.ones(coord_out.size-inds2.size)],axis=0)
        #out        = np.nan*np.ones_like(coord_out)
        #out[inds2] = np.interp(coord_out, coord_in[inds], var[inds])[inds2] #, left=np.nan, right=np.nan) #this produces weird results close to bottom
        #return out
    else:
        return np.nan*np.ones_like(coord_out)

def interp_sigma2depth_v2(var, coord_in, coord_out):
    '''slightly different formulation - doesn't work for some reason'''
    coord_in2,inds=np.unique(coord_in,return_index=True)
    if len(inds)>1:
        return np.interp(coord_out, coord_in2, var[inds], left=np.nan, right=np.nan)
    else:
        return np.nan*np.ones_like(coord_out)
    
def sigma2depth(var, coord_in, coord_out, dim='sigma',dim_out='depth',loop_over=None,loop_step=None,v2=False,spill=False,dumpath='/scratch/anu074/'):
    '''Interpolate var from sigma space to depth space (could be some other coordinate transformation as well)
    :var, xr.DataArray (or dask-array): assumed to have dims [x,y,dim] 
    :coord_in, xr.DataArray (or dask-array): assumed to have dims [x,y,dim]
    :coord_out, 1D numpy-array: output coordinate
    :dim, str: dimension over which the interpolation will take place
    :dim_out, str: name of the output dimension
    '''
    if loop_over==None:
        if v2:
            return xr.apply_ufunc(
                interp_sigma2depth_v2, var, coord_in,
                input_core_dims=[[dim], [dim]],
                output_core_dims=[[dim_out]],
                output_sizes={dim_out:coord_out.size},
                kwargs={'coord_out':coord_out},
                vectorize=True,
                dask='parallelized',
                output_dtypes=[var.dtype]
                )
        else:
            return xr.apply_ufunc(
                interp_sigma2depth, var, coord_in,
                input_core_dims=[[dim], [dim]],   # dim indicates the dimension over which we don't want to broadcast
                output_core_dims=[[dim_out]],
                output_sizes={dim_out:coord_out.size},
                kwargs={'coord_out':coord_out},
                vectorize=True, # !Important!
                dask='parallelized',
                output_dtypes=[var.dtype]
                )
    else:
        if v2:
            for j in range(0,var[loop_over].size,loop_step):
                out0 = xr.apply_ufunc(
                interp_sigma2depth_v2, var.isel({loop_over:slice(j,j+loop_step)}), coord_in.isel({loop_over:slice(j,j+loop_step)}),
                input_core_dims=[[dim], [dim]],
                output_core_dims=[[dim_out]],
                output_sizes={dim_out:coord_out.size},
                kwargs={'coord_out':coord_out},
                vectorize=True,
                dask='parallelized',
                output_dtypes=[var.dtype]
                ).compute()
                if j==0:
                    mask = out0.notnull().sum(dim=loop_over).compute()
                    out  = out0.sum(dim=loop_over).compute()
                else:
                    mask = mask	+ out0.notnull().sum(dim=loop_over).compute()
                    out  = out  + out0.sum(dim=loop_over).compute()
            #
            return out/mask
        else:
            for j in range(0,var[loop_over].size,loop_step):
                out0 = xr.apply_ufunc(
                interp_sigma2depth, var.isel({loop_over:slice(j,j+loop_step)}), coord_in.isel({loop_over:slice(j,j+loop_step)}),
                input_core_dims=[[dim], [dim]],
                output_core_dims=[[dim_out]],
                output_sizes={dim_out:coord_out.size},
                kwargs={'coord_out':coord_out},
                vectorize=True,
                dask='parallelized',
                output_dtypes=[var.dtype]
                ).compute()
                if j==0:
                    mask = out0.notnull().sum(dim=loop_over).compute()
                    out  = out0.sum(dim=loop_over).compute()
                else:
                    mask = mask + out0.notnull().sum(dim=loop_over).compute()
                    out  = out  + out0.sum(dim=loop_over).compute()
            #
            return out/mask

def interp_depth2sigma(var, coord_out, coord_in):
    '''
    This is the inverse of interp_sigma2depth
    var :: vector
    coord_out :: vector
    coord_in ::    
    '''
    inds  = np.where(np.isfinite(var))[0]
    if len(inds)>1:
        return np.interp(coord_out, coord_in[inds], var[inds], left=np.nan, right=np.nan)
    else:
        return np.nan*np.ones_like(coord_out)

def depth2sigma(var, coord_in, coord_out, dim='depth',dim_out='sigma',loop_over=None,loop_step=None,spill=False,dumpath='/scratch/anu074/'):
    '''Interpolate a variable (var) from a regular depth grid (coord_in) to depth of given sigma surfaces (irregular depth grid - coord_out)'''
    return xr.apply_ufunc(
        interp_depth2sigma, var, coord_out,
        input_core_dims=[[dim], [dim_out]],   # dim indicates the dimension over which we don't want to broadcast
        output_core_dims=[[dim_out]],
        output_sizes={dim_out:coord_out[dim_out].size},
        kwargs={'coord_in':coord_in},
        vectorize=True,                                                                                                                                        
        dask='parallelized',
        output_dtypes=[var.dtype]
        )

def eddy_v_bflx(vvel,dens,depth_in,depth_out,vvel_z,dens_z,loop_over='time',loop_step=20,g=9.81):
    '''
    Calculate the eddy buoyancy flux
    #
    vvel      :: y-velocity in sigma coordinates
    dens      :: in-situ density in sigma coordinates
    depth_in  :: layer depth
    depth_out :: desired output depth axis
    vvel_z    :: mean y-velocity in z coordinates
    dens_z    :: mean density in z coordinates
    loop_over :: dimension to loop over
    loop_step :: step size
    '''
    for j in range(0,vvel[loop_over].size,loop_step):
        vvel_dum = sigma2depth(vvel.isel({loop_over:slice(j,j+loop_step)}), depth_in.isel({loop_over:slice(j,j+loop_step)}), depth_out, dim='sigma',dim_out='depth')
        dens_dum = sigma2depth(dens.isel({loop_over:slice(j,j+loop_step)}), depth_in.isel({loop_over:slice(j,j+loop_step)}), depth_out, dim='sigma',dim_out='depth')
        #
        vvel_dum = vvel_dum.rename({'depth':'depth_out'}).assign_coords(depth_out=depth_out)
        dens_dum = dens_dum.rename({'depth':'depth_out'}).assign_coords(depth_out=depth_out)
        #
        dum,vsprime = noresm2scalar(vvel_dum,vvel_dum-vvel_z)
        dens_prime = dens_dum-dens_z
        # all the variables are again on z-levels so we can simply sum up and divide by the timeseries length at the end.
        if j==0:
            out=(vsprime*dens_prime).sum(dim=loop_over).compute()
        else:
            out=out+(vsprime*dens_prime).sum(dim=loop_over).compute()
        #
    return (-g*(out/vvel[loop_over].size)/dens_z).rename('eddy_v_bflx')

def eddy_vz_flx(vvel,z,depth_out,loop_over='time',loop_step=20):
    '''
    Calculate <v'z'> on sigma coordinates and interpolate to z-levels 
    #
    vvel      :: y-velocity in sigma coordinates
    z         :: layer depth (not thickness) in sigma coordinates
    depth_in  :: layer depth 
    depth_out :: desired output depth axis
    loop_over :: dimension to loop over
    loop_step :: step size
    '''
    #mask0=vvel.notnull()
    vvel=vvel.fillna(0)
    for j in range(0,vvel[loop_over].size,loop_step):
        if j==0:
            #mask   = mask0.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
            vmean  = vvel.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
            zmean  = z.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute() 
        else:
            #mask   = mask  + mask0.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
            vmean  = vmean + vvel.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
            zmean  = zmean + z.isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
    #
    vmean = vmean/vvel[loop_over].size
    zmean = zmean/vvel[loop_over].size #depth axis will always have a value
    # make a zonal mean here - doesn't really matter
    dumprime,vsprime = noresm2scalar(vvel-vmean,vvel-vmean) #.mean(dim='x'))
    zprime = z-zmean #.mean(dim='x')
    #
    vsprime = vsprime.fillna(0)
    for j in range(0,vvel[loop_over].size,loop_step):
        if j==0:
            vz   = (vsprime*zprime).isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
        else:
            vz   = vz + (vsprime*zprime).isel({loop_over:slice(j,j+loop_step)}).sum(dim=loop_over).compute()
    #
    vz   = vz/vvel[loop_over].size
    vz_z = sigma2depth(vz, zmean, depth_out, dim='sigma',dim_out='depth')
    #
    return vz.rename('eddy_vdz_flx'), zmean.rename('zmean'), vz_z.rename('eddy_vdz_flx_z')


def apply_deplhi(p,t,zbottom,s,sdim):
    '''                                                                                                                                             
    Calculate geopotential at layer interfaces.
    Note the unit conversion!

    Input

    p    :: pressure array in [Pa] (float)
    t    :: temperature array in [deg C] (float)
    s    :: salinity array in [g/kg] (float)
    sdim :: length of sigma dimension (int)
    
    Output
    
    phi  :: interface geopotential (negative down) in [cm^2/s^2]
    '''
    p         = np.concatenate([np.zeros(1),p])*10 #g/cm/s^2
    phi       = np.zeros(sdim+1) #phi
    phi[-1]   = -980.6*zbottom*1.E2 #needs to be in cm
    #
    tnew     = t.copy()
    nanind   = np.where(np.isnan(t))[0]
    for k in nanind:
        tnew[k] = tnew[k-1]
    #
    for k in range(sdim)[::-1]:
        if (p[k+1]-p[k])<1:
           phi[k]=phi[k+1]
        else:
           dphi,alpu,alpl = eosben07_delphi(p[k],p[k+1],tnew[k],s) #aplha is specific volume of the upper and lower layer
           phi[k] = phi[k+1]-dphi
    #
    return phi

def pgfx(p,dpu,pu,pdum_u,t,tdum_u,phi,phi_u,s,sdim,dx):
    '''
    Calculate the pressure gradient force in x-direction.
    Note that the pressure is an interface value.

    Input

    p      :: interface pressure array in [Pa] (float)
    dpu    :: layer pressure thickness at u-point [Pa] (float)
    pu     :: interface pressure array at u-pount [Pa] (float)
    pdum_u :: interface pressure array at p-point shifted to the left
              i.e index i on pdum_u corresponds to i-1 on p [Pa] (float)
    t      :: layer temperature array in [deg C] (float)
    tdum_u :: layer temperature array shifted to the left [deg C] (float)
    phi    :: interface geopotential height (negative down) [cm^2/s]
    phi_u  :: interface geopotential height shifted to the left
              (negative down) [cm^2/s] (float)
    s      :: salinity array in [g/kg] (float)
    sdim   :: length of sigma dimension (int)
    dx     :: cell width in x-direction [m] (float)
    '''
    pgfx    = np.zeros(sdim)
    tnew    = t.copy()
    tnew_u  = tdum_u.copy()
    #
    sc     = 10 # convert pressure from Pa to g/cm/s^2
    # add zero surface pressure
    p      = np.concatenate([np.zeros(1),p])*sc
    pu     = np.concatenate([np.zeros(1),pu])*sc
    pdum_u = np.concatenate([np.zeros(1),pdum_u])*sc
    dpu    = dpu*sc
    #
    prs = pu[1:]-.5*dpu
    #
    nanind   = np.where(np.isnan(t))[0]
    nanind_u = np.where(np.isnan(tdum_u))[0]
    #tnew[nanind]     = tnew[nanind-1]
    #tnew_u[nanind_u] = tnew_u[nanind-1]
    #
    for k in nanind:
        tnew[k] = tnew[k-1]
    for k in nanind_u:
        tnew_u[k] = tnew_u[k-1]
    #
    for k in range(sdim)[::-1]:
        #
        if p[-1]<=prs[k]:
            kup = sdim-1
        else:
            kup = np.where(p>prs[k])[0][0]-1
        #
        if pdum_u[-1]<=prs[k]:
            kum = sdim-1
        else:
            kum = np.where(pdum_u>prs[k])[0][0]-1
        #
        dphip,alpup,alplp=eosben07_delphi(prs[k],p[kup+1], \
                      tnew[kup],s)
        #
        dphim,alpum,alplm=eosben07_delphi(prs[k],pdum_u[kum+1], \
                      tnew_u[kum],s)
        #
        phi_p   = phi[kup+1]-dphip
        phi_m   = phi_u[kum+1]-dphim
        pgfx[k] = -(phi_p-phi_m)
    #
    # remove barotropic gradient - we want to keep this
    #
    # pgfx = pgfx-np.nansum(pgfx*dpu)/np.nansum(dpu) #pbu_p
    # pgfx = pgfx-pgfxm/pbu_p
    #
    return pgfx/(1E4*dx) #1E4 to convert cm^2/s to m^2/s
    

def pgfy(p,dpv,pv,pdum_v,t,tdum_v,phi,phi_v,s,sdim,dy):
    '''
    Calculate the pressure gradient force in x-direction
    Note that the pressure is an interface value.
    
    Input
    
    p      :: interface pressure array in [Pa] (float)
    dpv    :: layer pressure thickness at v-point [Pa] (float)
    pv     :: interface pressure array at v-pount [Pa] (float)
    pdum_v :: interface pressure array at p-point shifted down
              i.e index j on pdum_v corresponds to j-1 on p [Pa] (float)
    t      :: layer temperature array in [deg C] (float)
    tdum_v :: layer temperature array shifted down [deg C] (float)
    phi    :: interface geopotential height (negative down) [cm^2/s]
    phi_v  :: interface geopotential height shifted down
              (negative down) [cm^2/s] (float)
    s      :: salinity array in [g/kg] (float)
    sdim   :: length of sigma dimension (int)
    dy     :: cell width in y-direction [m] (float)
    '''
    pgfy    = np.zeros(sdim)
    tnew    = t.copy()
    tnew_v  = tdum_v.copy()
    #
    sc     = 10 #convert pressure units to g/cm/s^2
    # add zero surface pressure
    p      = np.concatenate([np.zeros(1),p])*sc
    pv     = np.concatenate([np.zeros(1),pv])*sc
    pdum_v = np.concatenate([np.zeros(1),pdum_v])*sc
    dpv    = dpv*sc
    #
    prs = pv[1:]-.5*dpv
    #
    nanind   = np.where(np.isnan(t))[0]
    nanind_v = np.where(np.isnan(tdum_v))[0]
    for k in nanind:
        tnew[k] = tnew[k-1]
    for k in nanind_v:
        tnew_v[k] = tnew_v[k-1]
    #
    for k in range(sdim)[::-1]:
        #
        if p[-1]<=prs[k]:
            kvp = sdim-1
        else:
            kvp = np.where(p>prs[k])[0][0]-1
        #
        if pdum_v[-1]<=prs[k]:
            kvm = sdim-1
        else:
            kvm = np.where(pdum_v>prs[k])[0][0]-1
        #
        dphip,alpup,alplp=eosben07_delphi(prs[k],p[kvp+1], \
                      tnew[kvp],s)
        #
        dphim,alpum,alplm=eosben07_delphi(prs[k],pdum_v[kvm+1], \
                      tnew_v[kvm],s)
        #
        phi_p   = phi[kvp+1]-dphip
        phi_m   = phi_v[kvm+1]-dphim
        pgfy[k] = -(phi_p-phi_m)
    #
    # remove barotropic gradient - we want to keep this
    #
    # pgfy = pgfy-np.nansum(pgfy*dpv)/np.nansum(dpv) #pbv_p
    # pgfy = pgfy-pgfym/pbv_p
    #
    return pgfy/(1E4*dy) #

def calculate_pgforc(data,dx_m=2E3,S=35,return_pgfx=True,return_pgfy=True):
    """
    """
    ydim = data.y.size
    xdim = data.x.size
    sdim = data.sigma.size
    #
    dp = data.dp.assign_coords(sigma=np.arange(sdim)).chunk(chunks={'time':1})
    T  = data.temp.assign_coords(sigma=np.arange(sdim)).chunk(chunks={'time':1,'x':xdim})
    if S==None:
        S = data.saln.assign_coords(sigma=np.arange(sdim)).chunk(chunks={'time':1,'x':xdim})
    #
    p  = dp.chunk(chunks={'time':1,'x':xdim}).cumsum(dim='sigma')
    #
    zbottom   = data.dz.chunk(chunks={'time':1,'x':-1}).sum(dim='sigma')-data.sealv.chunk(chunks={'time':1,'x':-1})
    #
    phi = xr.apply_ufunc(
    apply_deplhi, p, T, zbottom,
    input_core_dims=[['sigma'], ['sigma'],[]],
    output_core_dims=[['sigma2']],
    kwargs={'s':S,'sdim':sdim},
    vectorize=True,
    dask='parallelized',
    output_dtypes=[p.dtype],
    dask_gufunc_kwargs={'output_sizes':{'sigma2':sdim+1}}
    )
    #
    # these variables are defined to cover i-1, j-1 indices
    pdum_u = xr.concat([p.isel(x=slice(xdim-1,xdim)),p],dim='x').chunk(chunks={'time':1})
    pdum_v = xr.concat([p.isel(y=slice(ydim-1,ydim)),p],dim='y').chunk(chunks={'time':1})
    Tdum_u = xr.concat([T.isel(x=slice(xdim-1,xdim)),T],dim='x').chunk(chunks={'time':1})
    Tdum_v = xr.concat([T.isel(y=slice(ydim-1,ydim)),T],dim='y').chunk(chunks={'time':1})
    phi_u  = xr.concat([phi.isel(x=slice(xdim-1,xdim)),phi],dim='x').chunk(chunks={'time':1,'sigma2':sdim+1,'x':-1,'y':-1})
    phi_v  = xr.concat([phi.isel(y=slice(ydim-1,ydim)),phi],dim='y').chunk(chunks={'time':1,'sigma2':sdim+1,'x':-1,'y':-1})
    # p-variables need to be concantenated to calculate dpu, dpv
    pdum_u = xr.concat([xr.zeros_like(pdum_u.isel(sigma=slice(0,1))),pdum_u],dim='sigma').assign_coords(sigma=np.arange(sdim+1)).chunk(chunks={'sigma':sdim+1})
    pdum_v = xr.concat([xr.zeros_like(pdum_v.isel(sigma=slice(0,1))),pdum_v],dim='sigma').assign_coords(sigma=np.arange(sdim+1)).chunk(chunks={'sigma':sdim+1})
    #
    #########################
    # CALCULATE PU
    qu = xr.concat([pdum_u.isel(x=slice(1,xdim+1),sigma=-1),pdum_u.isel(x=slice(0,xdim),sigma=-1)],dim='dum',join='right').min('dum').drop('sigma').expand_dims(dim={'sigma':np.arange(sdim)},axis=1)
    #
    dpu = .5* \
    ((xr.concat([qu,pdum_u.isel(x=slice(0,xdim),sigma=slice(1,sdim+1)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum') \
    - xr.concat([qu,pdum_u.isel(x=slice(0,xdim),sigma=slice(0,sdim)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum')) \
    +(xr.concat([qu,pdum_u.isel(x=slice(1,xdim+1),sigma=slice(1,sdim+1)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum') \
    - xr.concat([qu,pdum_u.isel(x=slice(1,xdim+1),sigma=slice(0,sdim)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum'))) #],dim='dum').mean('dum')
    #
    pu = dpu.chunk(chunks={'time':1,'sigma':sdim,'x':xdim,'y':ydim//2}).cumsum('sigma')
    ########################
    #CALCULATE PV
    qv = xr.concat([pdum_v.isel(y=slice(1,ydim+1),sigma=-1),pdum_v.isel(y=slice(0,ydim),sigma=-1)],dim='dum',join='right').min('dum').drop('sigma').expand_dims(dim={'sigma':np.arange(sdim)},axis=1)
    #
    dpv = 0.5* \
    ((xr.concat([qv,pdum_v.isel(y=slice(0,ydim),sigma=slice(1,sdim+1)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum') \
    -xr.concat([qv,pdum_v.isel(y=slice(0,ydim),sigma=slice(0,sdim)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum')) \
    +(xr.concat([qv,pdum_v.isel(y=slice(1,ydim+1),sigma=slice(1,sdim+1)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum') \
    -xr.concat([qv,pdum_v.isel(y=slice(1,ydim+1),sigma=slice(0,sdim)).assign_coords({'sigma':np.arange(sdim)})],'dum',join='left').min('dum'))) #] ,dim='dum').mean('dum')
    #
    pv = dpv.chunk(chunks={'time':1,'sigma':sdim,'x':xdim//2,'y':ydim}).cumsum('sigma').rename('pv')
    #
    #
    # cut for i-1
    pdum_u2 = pdum_u.isel(x=slice(0,xdim),sigma=slice(1,sdim+1)).chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1,'y':ydim}).assign_coords(sigma=np.arange(sdim))
    Tdum_u2 = Tdum_u.isel(x=slice(0,xdim)).chunk(chunks={'time':1,'y':-1,'x':-1,'y':ydim})
    phi_u2  = phi_u.isel(x=slice(0,xdim)).chunk(chunks={'time':1,'y':-1,'x':-1,'y':ydim})
    #
    # cut for j-1
    pdum_v2 = pdum_v.isel(y=slice(0,ydim),sigma=slice(1,sdim+1)).chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1}).assign_coords(sigma=np.arange(sdim))
    Tdum_v2 = Tdum_v.isel(y=slice(0,ydim)).chunk(chunks={'time':1,'y':-1,'x':-1})
    phi_v2  = phi_v.isel(y=slice(0,ydim)).chunk(chunks={'time':1,'y':-1,'x':-1})
    #
    # rechunk variables for dp/dx
    p2      = p.chunk(chunks={'time':1,'sigma':sdim,'y':ydim,'x':-1})
    dpu2    = dpu.chunk(chunks={'time':1,'sigma':sdim,'y':ydim,'x':-1})
    pu2     = pu.chunk(chunks={'time':1,'sigma':sdim,'y':ydim,'x':-1})
    T2      = T.chunk(chunks={'time':1,'sigma':sdim,'y':ydim,'x':-1})
    phi2    = phi.chunk(chunks={'time':1,'sigma2':sdim+1,'y':ydim,'x':-1})
    #
    # rechunk variables for dp/dy
    p3      = p.chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1})
    dpv3    = dpv.chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1})
    T3      = T.chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1})
    phi3    = phi.chunk(chunks={'time':1,'sigma2':sdim+1,'y':-1,'x':-1})
    pv2     = pv.chunk(chunks={'time':1,'sigma':sdim,'y':-1,'x':-1})
    #    
    #
    pgfx_all = xr.apply_ufunc(
        pgfx, p2, dpu2, pu2, pdum_u2, T2, Tdum_u2, phi2, phi_u2,
        input_core_dims=[['sigma'], ['sigma'], ['sigma'], ['sigma'], ['sigma'],['sigma'], ['sigma2'], ['sigma2']],
        output_core_dims=[['sigma']],
        kwargs={'s':S,'sdim':sdim,'dx':dx_m},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[p.dtype],
        dask_gufunc_kwargs={'output_sizes':{'sigma':sdim}}
        )
    #
    pgfy_all = xr.apply_ufunc(
        pgfy, p3, dpv3, pv2, pdum_v2, T3, Tdum_v2, phi3, phi_v2,
        input_core_dims=[['sigma'], ['sigma'], ['sigma'], ['sigma'], ['sigma'],['sigma'], ['sigma2'], ['sigma2']],
        output_core_dims=[['sigma']],
        kwargs={'s':S,'sdim':sdim},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[p.dtype],
        dask_gufunc_kwargs={'output_sizes':{'sigma':sdim}}
        )
    #
    if return_pgfx and not return_pgfy:
        return pgfx_all.rename('pgfx')
    elif return_pgfy and not return_pgfx:
        return pgfy_all.rename('pgfy')
    else:
        return pgfx_all.rename('pgfx'), pgfx_all.rename('pgfy')

def rossby_radius(dens_z,depth,dim='depth',f=1E-4,g=9.81):
    '''Calculate Rossby radius of derormation as a integral of the buoancy frequency '''
    dz_mid = np.gradient(depth)    
    N2 = (g/dens_z)*dens_z.differentiate(dim)/dz_mid
    N2.where(N2>0, other=0)
    return ((np.sqrt(N2)*dz_mid).sum(dim=dim)/(np.pi*f)).rename('rossby_radius')

def gaussian(x_fwhm, y_fwhm, cutoff=2, amplitude=1, theta=0, mfac=4*2*np.log(2)):
    """
    Two dimensional Gaussian function - here defined in terms of
    full width half maximum (i.e not std), otherwise we are
    following astropy.
    # mfac = 4*2*np.log(2) # full width, half maximum
    # mfact = 0.5          # std
    #
    # although cutoff of 3 is usually suggested, here cutoff of 2 seems to be adequate                                                                                                                            
    See astropy.modeling.functional_models - gaussian
    """
    #
    #
    x = np.arange(0, x_fwhm*cutoff+1, 1, float)
    y = np.arange(0, y_fwhm*cutoff+1, 1, float)
    x,y = np.meshgrid(x,y)
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    x_fwhm2 = x_fwhm**2
    y_fwhm2 = y_fwhm**2
    xdiff = x - x_fwhm*cutoff//2
    ydiff = y - y_fwhm*cutoff//2
    a = mfac * ((cost2 / x_fwhm2) + (sint2 / y_fwhm2))
    b = mfac * ((sin2t / x_fwhm2) - (sin2t / y_fwhm2))
    c = mfac * ((sint2 / x_fwhm2) + (cost2 / y_fwhm2))
    return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))     

def animate(frame):
    global data,image,var
    image.set_array(data[var].isel(time=frame).values)
    return image

def update(j):
    ax.set_title(j)
    dum = data.sst.isel(time=j+1)

def eosben07_const():
    '''
    EOS constants in SI units
    '''
    a11= 9.9985372432159340e+02
    a12= 1.0380621928183473e+01
    a13= 1.7073577195684715e+00  
    a14=-3.6570490496333680e-02
    a15=-7.3677944503527477e-03
    a16=-3.5529175999643348e-03
    a21= 1.0  
    a22= 1.0316374535350838e-02
    a23= 8.9521792365142522e-04
    a24=-2.8438341552142710e-05
    a25=-1.1887778959461776e-05
    a26=-4.0163964812921489e-06
    b11= 1.7083494994335439e-02
    b12= 7.1567921402953455e-05
    b13= 1.2821026080049485e-05
    b21= 1.1995545126831476e-05
    b22= 5.5234008384648383e-08
    b23= 8.4310335919950873e-09
    #
    return a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23

def eosben07_const2(pref=2000):
    '''
    Secondary constants
    '''
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23 = eosben07_const()
    ap21=a21+b21*pref
    ap22=a22+b22*pref
    ap23=a23+b23*pref
    ap24=a24
    ap25=a25
    ap26=a26
    ap11=a11+b11*pref-ap21
    ap12=a12+b12*pref-ap22
    ap13=a13+b13*pref-ap23
    ap14=a14-ap24
    ap15=a15-ap25
    ap16=a16-ap26
    ap210=a21
    ap220=a22
    ap230=a23
    ap240=a24
    ap250=a25
    ap260=a26
    ap110=a11-ap210
    ap120=a12-ap220
    ap130=a13-ap230
    ap140=a14-ap240
    ap150=a15-ap250
    ap160=a16-ap260
    return ap11,ap12,ap13,ap14,ap15,ap16,ap21,ap22,ap23,ap24,ap25,ap26,ap110,ap120,ap130,ap140,ap150,ap160,ap210,ap220,ap230,ap240,ap250,ap260

def eosben07_const_f():
    """
    EOS constants in BLOM units (cm)
    """
    a11= 9.9985372432159340e-01
    a12= 1.0380621928183473e-02
    a13= 1.7073577195684715e-03
    a14=-3.6570490496333680e-05
    a15=-7.3677944503527477e-06
    a16=-3.5529175999643348e-06
    b11= 1.7083494994335439e-10
    b12= 7.1567921402953455e-13
    b13= 1.2821026080049485e-13
    a21= 1.0                   
    a22= 1.0316374535350838e-02
    a23= 8.9521792365142522e-04
    a24=-2.8438341552142710e-05
    a25=-1.1887778959461776e-05
    a26=-4.0163964812921489e-06
    b21= 1.1995545126831476e-10
    b22= 5.5234008384648383e-13
    b23= 8.4310335919950873e-14
    return a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23

def eosben07_const_f2(pref=2000.e5):
    """
    Secondary constants
    """
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23 = eosben07_const_f()
    #
    ap21=a21+b21*pref
    ap22=a22+b22*pref
    ap23=a23+b23*pref
    ap24=a24
    ap25=a25
    ap26=a26
    ap11=a11+b11*pref-ap21
    ap12=a12+b12*pref-ap22
    ap13=a13+b13*pref-ap23
    ap14=a14-ap24
    ap15=a15-ap25
    ap16=a16-ap26
    ap210=a21
    ap220=a22
    ap230=a23
    ap240=a24
    ap250=a25
    ap260=a26
    ap110=a11-ap210
    ap120=a12-ap220
    ap130=a13-ap230
    ap140=a14-ap240
    ap150=a15-ap250
    ap160=a16-ap260
    #
    return ap11,ap12,ap13,ap14,ap15,ap16,ap21,ap22,ap23,ap24,ap25,ap26,ap110,ap120,ap130,ap140,ap150,ap160,ap210,ap220,ap230,ap240,ap250,ap260

def p_alpha(p,p0,th,s):
    """1/rho at pressure p and reference pressure p"""
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23=eosben07_const()
    #
    b1i=1/(b11+b12*th+b13*s)
    a1=(a11+th*(a12+a14*th+a15*s)+s*(a13+a16*s))*b1i
    a2=(a21+th*(a22+a24*th+a25*s)+s*(a23+a26*s))*b1i
    b2=(b21+b22*th+b23*s)*b1i
    #
    r=b2*(p-p0)+(a2-a1*b2)*np.log((a1+p)/(a1+p0))
    #
    return r

def eosben07_rho(p,th,s):
    """in-situ density from pressure, potential temperature and salinity"""
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23=eosben07_const()
    #
    r=(a11+th*(a12+a14*th+a15*s)+s*(a13+a16*s)+p*(b11+b12*th+b13*s))/(a21+th*(a22+a24*th+a25*s)+s*(a23+a26*s)+p*(b21+b22*th+b23*s))
    #
    return r

def eosben07_sig(p,th,s,pref=2000):
    """Potential density from pressure, potential temperature and salinity"""
    ap11,ap12,ap13,ap14,ap15,ap16,ap21,ap22,ap23,ap24,ap25,ap26,ap110,ap120,ap130,ap140,ap150,ap160,ap210,ap220,ap230,ap240,ap250,ap260 = eosben07_const2(pref=pref)
    #
    sig=(ap11+(ap12+ap14*th+ap15*s)*th+(ap13+ap16*s)*s)/(ap21+(ap22+ap24*th+ap25*s)*th+(ap23+ap26*s)*s)
    #
    return sig

def eosben07_sig0(p,th,s,pref=2000):
    """Potential density referenced to 0 from pressure, potential temperature, and salinity"""
    ap11,ap12,ap13,ap14,ap15,ap16,ap21,ap22,ap23,ap24,ap25,ap26,ap110,ap120,ap130,ap140,ap150,ap160,ap210,ap220,ap230,ap240,ap250,ap260 = eosben07_const2(pref=pref)
    #
    sig0=(ap110+(ap120+ap140*th+ap150*s)*th+(ap130+ap160*s)*s)/(ap210+(ap220+ap240*th+ap250*s)*th+(ap230+ap260*s)*s)
    #
    return sig0

def eosben07_rho_th(p,th,s):
    """ Explanation of the function """
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23=eosben07_const()
    #
    P=a11+th*(a12+a14*th+a15*s)+s*(a13+a16*s)+p*(b11+b12*th+b13*s)
    Qi=1/(a21+th*(a22+a24*th+a25*s)+s*(a23+a26*s)+p*(b21+b22*th+b23*s))
    #
    r=Qi*(a12+2.0*a14*th+a15*s+b12*p-Qi*P*(a22+2.0*a24*th+a25*s+b22*p))
    #
    return r

def eosben07_rho_s(p,th,s):
    """ Explanation of the function """
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23=eosben07_const()
    #
    P=a11+th*(a12+a14*th+a15*s)+s*(a13+a16*s)+p*(b11+b12*th+b13*s);
    Qi=1/(a21+th*(a22+a24*th+a25*s)+s*(a23+a26*s)+p*(b21+b22*th+b23*s))
    #
    r=Qi*(a13+a15*th+2.0*a16*s+b13*p-Qi*P*(a23+a25*th+2.0*a26*s+b23*p))
    #
    return r

def eosben07_tofsig(sg,s,pref=2000.e5):
    """
    invert temperature from sigma and salinity
    """
    ap11,ap12,ap13,ap14,ap15,ap16,ap21,ap22,ap23,ap24,ap25,ap26,ap110,ap120,ap130,ap140,ap150,ap160,ap210,ap220,ap230,ap240,ap250,ap260 = eosben07_const_f2(pref=pref)
    sg = sg*1E-3
    #print(ap14,ap24,ap12,ap21,ap22,ap16,ap26)
    a  = ap14-ap24*sg
    b  = ap12-ap22*sg+(ap15-ap25*sg)*s
    c  = ap11-ap21*sg+(ap13-ap23*sg+(ap16-ap26*sg)*s)*s
    return (-b-np.sqrt(b*b-4.*a*c))/(2.*a)

def eosben07_delphi(p1,p2,th,s):
    """
    Integrate specific volume with respect to pressure to find the
    difference in geopotential between two pressure levels in BLOM units
    """
    a11,a12,a13,a14,a15,a16,a21,a22,a23,a24,a25,a26,b11,b12,b13,b21,b22,b23 = eosben07_const_f()
    #
    r1_3=1./3.
    r1_5=1./5.
    r1_7=1./7.
    r1_9=1./9.
    #
    a1=a11+(a12+a14*th+a15*s)*th+(a13+a16*s)*s
    a2=a21+(a22+a24*th+a25*s)*th+(a23+a26*s)*s
    b1=b11+b12*th+b13*s
    b2=b21+b22*th+b23*s
    #c
    #c --- the analytic solution of the integral is
    #c       dphi=-(b2*(p2-p1)
    #c             +(a2-a1*b2/b1)*log((a1+b1*p2)/(a1+b1*p1)))/b1
    #c --- a truncated series expansion of the integral is used that provide
    #c --- better computational efficiency and accuarcy for most relevant
    #c --- parameters
    #c
    pm=.5*(p2+p1)
    r=.5*(p2-p1)/(a1+b1*pm)
    q=b1*r
    qq=q*q
    #
    dphi=-2.*r*(a2+b2*pm \
         +(a2-a1*b2/b1)*qq*(r1_3+qq*(r1_5+qq*(r1_7+qq*r1_9))))
    #
    alp1=(a2+b2*p1)/(a1+b1*p1)
    alp2=(a2+b2*p2)/(a1+b1*p2)
    #  
    return dphi, alp1, alp2
