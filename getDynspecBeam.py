import numpy as np
from astropy.coordinates import ITRS, EarthLocation, AltAz
import astropy.units as u
from astropy.constants import c

from lofarimaging import singlestationutil as stu
import casacore.tables as pt

import lofarantpos.geo as lofargeo
from lofarantpos.db import LofarAntennaDatabase

#based on code by Maaijke Mevius orginal code https://github.com/maaijke/beam_scripts

mydb = LofarAntennaDatabase()
LV614lba = mydb.phase_centres["LV614LBA"]
etrs_to_pqr = mydb.pqr_to_etrs["LV614LBA"].T

def get_global_pqr(station_name):
    phase_centre = mydb.phase_centres[station_name]
    return etrs_to_pqr @ (phase_centre - LV614lba)

def get_missing_elements(MSname, station):
    myt = pt.table(f"{MSname}/LOFAR_ANTENNA_FIELD", ack=False)
    flags = myt.getcol('ELEMENT_FLAG')
    antnames = list( pt.table( f"{MSname}/ANTENNA", ack=False).getcol( "NAME" ) )
    try:
        sortedflags =  flags[ antnames.index(station) ]
    except:
        print("finding flags fails, maybe non matching stations", station, antnames)
        sortedflags = np.ones( flags[0].shape, dtype=bool )
    return sortedflags


def getAzelBeam( antpos, altaz, itrf_to_enu, ref_pos ) :
    '''returns distance vector for all antennas and all altaz directions.
    Input:
    antpos: array (Nant x 3) ENU positions of all antennas
    altaz: array (AltAz)
    itrf_to_enu: np.array (3x3) Matrix to transform from itrf to ENU (pqr)
    Output:
    distance: np.array(Nant x Ndirections) relative distance vector for all directions
    '''
    itrf_dir = altaz.transform_to( ITRS () )
    enu_dir = lofargeo.transform(
        itrf_dir.cartesian.xyz.value.T,
        np.array( [ [ ref_pos.x.to(u.m).value, 
                      ref_pos.y.to(u.m).value, 
                      ref_pos.z.to(u.m).value ] ] ),
        itrf_to_enu
    )
    return antpos @ enu_dir.T


def getRaDecBeam( antpos, source, time, itrf_to_enu, ref_pos ):
    '''returns distance vector for all antennas and all times for a single direction.
    Input:
    antpos: array (Nant x 3) ENU positions of all antennas
    source: SkyCoord
    time: array of Time
    itrf_to_enu: np.array (3x3) Matrix to transform from itrf to ENU (pqr)
    ref_pos: antenna to use as ref position
    Output:
    distance: np.array(Nant x Ntimes) relative distance vector for all directions
    '''
    altaz = source.transform_to(
        AltAz(
            obstime=time,
            location=ref_pos
        )
        )
    itrf_dir = altaz.transform_to( ITRS () )
    enu_dir = lofargeo.transform(
        itrf_dir.cartesian.xyz.value.T,
        np.array( [ [ ref_pos.x.to(u.m).value, 
                      ref_pos.y.to(u.m).value, 
                      ref_pos.z.to(u.m).value ] ] ),
        itrf_to_enu
    )
    return antpos @ enu_dir.T


def getPower( distance_dir, distance_phase_center, freqs, common_axis=None ):
    '''Calculate power of beam response given phase_center and direction distance vector for all freqs'''
    if common_axis :
        nu = np.reshape( freqs, ( -1, 1, 1 ) )
        dd = np.reshape(
            distance_dir,
            ( 1, ) + distance_dir.shape #+ ( 1, )
        )
        dph = np.reshape(
            distance_phase_center,
            ( 1, ) + distance_phase_center.shape #+ ( 1, )
            )
    else:
        nu = np.reshape( freqs, ( -1, 1, 1, 1 ) )
        dd = np.reshape(
            distance_dir,
            ( 1, 1 ) + distance_dir.shape #+ ( 1, )
        )
        dph = np.reshape(
            distance_phase_center,
            ( 1, ) + distance_phase_center.shape[:-1] + ( 1, -1 )
            )
        
    A = np.sum ( np.exp( ( -2 * np.pi * 1.j * nu / c * ( dd - dph ) ).value ), axis=-1 )
    return np.abs(A)/dph.shape[-1]


def getDynspec( station, rcumode, radec, phasedir, times, freqs ) :
    antpos, rot_matrix = stu.get_station_xyz(
        station,
        rcumode,
        mydb
    )
    
    ref_pos = EarthLocation.from_geocentric(
        *mydb.phase_centres[station],
        unit=u.m
    )
    etrs_to_local_north = mydb.pqr_to_localnorth(station) @ mydb.pqr_to_etrs[station].T
    
    distance_phase_center = getRaDecBeam( antpos, phasedir, times, etrs_to_local_north, ref_pos )
    distance_dir = getRaDecBeam( antpos, radec, times, etrs_to_local_north, ref_pos )
    dynspec = getPower( distance_dir.T, distance_phase_center.T, freqs, common_axis=True )
    return dynspec, distance_phase_center, distance_dir

def getBeamPower( station, rcumode, azel, phasedir, times, freqs, MSname='') :
    antpos, rot_matrix = stu.get_station_xyz(
        station,
        rcumode,
        mydb
    )
    
    ref_pos = EarthLocation.from_geocentric(
        *mydb.phase_centres[station],
        unit=u.m
    )
    etrs_to_local_north = mydb.pqr_to_localnorth(station) @ mydb.pqr_to_etrs[station].T
    
    distance_phase_center = getRaDecBeam( antpos, phasedir, times, etrs_to_local_north, ref_pos )
    azel = AltAz(azel.az, azel.alt, location=ref_pos)
    distance_dir = getAzelBeam( antpos, azel, etrs_to_local_north, ref_pos )
    if not MSname=='':
        flagged_antennas = get_missing_elements(MSname, station)[:,0]
        if "CS" in station or 'RS' in station:
            flagged_antennas = flagged_antennas[ 48:]
    else:
        flagged_antennas = np.zeros( distance_dir.shape[0], dtype=bool )
    print(f"flagged {np.sum(flagged_antennas)} from {MSname}")
    good_ant = np.logical_not( flagged_antennas )[ :distance_dir.shape[0] ]  # because of stupid PL611
    dynspec = getPower( distance_dir[good_ant].T, distance_phase_center[good_ant].T, freqs, common_axis=False )
    return dynspec, distance_phase_center, distance_dir

