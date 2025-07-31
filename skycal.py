import numpy as np
#based on code by P.Kruger and ...
if True:
    from pygdsm import LFSMObserver as GSMObserver
    NSIDE=256


import healpy as hp
from astropy.time import Time


from astropy.utils.iers import conf
conf.auto_max_age = None

def sphere2cart(theta,phi):
    '''
    convert spherical coordinates to cartesian unit coordinates

    Parameters
    ----------
    theta : (array of) float
        polar angle theta (w.r.t. zenith) in rad
    phi : (array of) float
        azimuth angle phi in rad

    Returns
    -------
    array of float
         cartesian direction vectors S, dim.(3,) + shape (theta)   

    '''
    
    return np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)])



def get_lm(nside=NSIDE,plot=False):
    npix=hp.nside2npix(nside)
    XYZ=np.array(hp.pix2vec(nside,range(npix)))#.T
#    mask = XYZ[0]<0
    l=XYZ[1]
    m=XYZ[2]
    if plot:
        hp.orthview(l,half_sky = True, title = 'l')
        hp.orthview(m,half_sky = True, title = 'm')
    return l,m 

def get_coord(nside=NSIDE,plot=False):
    npix=hp.nside2npix(nside)
    XYZ=np.array(hp.pix2vec(nside,range(npix)))#.T
    mask = XYZ[0]<0
    t=XYZ[0].copy();XYZ[0]=XYZ[2];XYZ[2]=t #swap X and Z
    XYZ=hp.vec2ang(XYZ.T)
    az_rad = np.ma.masked_array(XYZ[1], mask = mask )
    z_rad = np.ma.masked_array(XYZ[0], mask = mask )
    if plot:
        hp.orthview(az_rad,title='az')
        hp.orthview(z_rad,title='zenith')
    return az_rad,z_rad


def get_sky(dates,freqs,latlonel):
    (latitude, longitude, height) = latlonel
    ov = GSMObserver()
    ov.lon = longitude
    ov.lat = latitude
    ov.elev = height
    skys = []
    # from ov: Rotation is quite slow, only recompute if time has changed, or it has never been run
    for date0 in dates:
        print ("adding",date0.datetime)
        skys.append([])
        for freq in freqs:
            print (f"adding{freq.to('MHz')}")
            power = ov.generate(freq.to("MHz").value,obstime=Time(date0))
            skys[-1].append(power)
            #ov.view(logged=True, title=str(freq.to("MHz")) + "_" + str(date0), show=False)
    del ov
    return skys
            
    
def interpolate(patt,z1,az1,tz1,ta1):
    '''includes some assumptions about the input z1,az1'''
    nz=len(z1)
    na=len(az1)
#    print(patt.shape,nz,na,tz1.shape,ta1.shape)
    patt=patt.reshape([nz*na])
    iz1=np.array(tz1,dtype='int')
    ia1=np.array(ta1/5,dtype='int')
    iz1-=iz1*(iz1<0)
    ia1-=ia1*(ia1<0)
    iz1-=iz1*(iz1>=nz-1)
    ia1-=ia1*(ia1>=na-1)
    wz=tz1-iz1
    wa=ta1/5-ia1
    it=iz1*na+ia1
    z3=(1-wz)*( (1-wa)*patt[it]+(wa)*patt[it+1] ) + (wz)*( (1-wa)*patt[it+na]+(wa)*patt[it+1+na] )
    return z3

def interpolate_beam(beam_pattern,zenith,azimuth):
    az_rad,z_rad = get_coord(plot=False)
    beamgrid = interpolateN(beam_pattern,zenith,azimuth,np.degrees(z_rad),np.degrees(az_rad))
    return beamgrid,az_rad,z_rad


def interpolateN(patt,z_in,az_in,z_out,az_out):
    '''z_in,az_in are the coordinates of patt. z_out,az_out are the gridded coordinates for the output array (healpy)'''
    nz=len(z_in)
    na=len(az_in)
    patt=patt.reshape(patt.shape[:-2]+(nz*na,)) #assume last two axes are coordinates
    #caclulate nearest index on the input _grid
    iz_grid = np.argmin(np.abs(z_out[:,np.newaxis] - z_in[np.newaxis]),axis=1)
    iaz_grid = np.argmin(np.abs(np.remainder(az_out[:,np.newaxis] - az_in[np.newaxis]+180,360)-180),axis=1)
    #weights
    wz=z_out-z_in[iz_grid]
    iz_grid[wz<0]-=1
    iz_grid[iz_grid<0]=0
    iz_grid[iz_grid>=(nz-1)]=nz-2
    wz=z_out-z_in[iz_grid]
    wnorm = wz - (z_out-z_in[iz_grid+1])
    wz/=wnorm
    iaz_grid[iaz_grid>=(na-1)]=0
    wa=np.remainder(az_out-az_in[iaz_grid]+180,360)-180
    iaz_grid[wa<0]-=1
    iaz_grid[iaz_grid<0]=na-2
    wa=np.remainder(az_out-az_in[iaz_grid]+180,360)-180
    wnorm = wa + np.remainder(az_in[iaz_grid+1]-az_out+180,360)-180
    wa/=wnorm
    #index
    it=iz_grid*na+iaz_grid
    z3=(1-wz)*( (1-wa)*patt[...,it]+(wa)*patt[...,it+1] ) + (wz)*( (1-wa)*patt[...,it+na]+(wa)*patt[...,it+1+na] )
    return z3

def CalcPower(freq,latlonel,dates,beamsky):
    (latitude, longitude, elevation) = latlonel
    Ntime=len(dates)
    pwr=np.zeros([2,Ntime])
    ov = GSMObserver()
    ov.lon = longitude
    ov.lat = latitude
    ov.elev = elevation
    for x,date0 in  tqdm(enumerate(dates)):
        ov.date = date0
        ov.generate(freq.to("MHz").value,obstime=Time(date0))    
        pwr[0,x]=np.sum( ov.observed_sky*beamsky[0])
        print ("power",x,pwr[:,x],beamsky.shape)
        pwr[1,x]=np.sum( ov.observed_sky*beamsky[1])
        print ("power",x,pwr[:,x])
    del ov;
    return pwr;

def CalcPowerN(freq,latlonel,dates,beamsky):
    Nlba=beamsky[0].shape[0];
    print(Nlba)
    (latitude, longitude, elevation) = latlonel
    Ntime=len(dates)
    pwr=np.zeros([Nlba,2,Ntime])
    ov = GSMObserver()
    ov.lon = longitude
    ov.lat = latitude
    ov.elev = elevation
    print(beamsky.shape,pwr.shape)
    for x,date0 in  tqdm(enumerate(dates)):
        print (x,date0)
        ov.date = date0
        ov.generate(freq.to("MHz").value,obstime=Time(date0))    
        for i in range(Nlba):
            pwr[i,0,x]=np.sum( ov.observed_sky*beamsky[0][i]/np.sum(beamsky[0][i]))
            pwr[i,1,x]=np.sum( ov.observed_sky*beamsky[1][i]/np.sum(beamsky[0][i]))
    del ov;
    return pwr;

