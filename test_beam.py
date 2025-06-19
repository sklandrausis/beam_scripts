from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord, FK5
from astropy.time import Time

import astropy.units as u
import numpy as np
from getDynspecBeam import *
from skycal import get_sky, interpolate_beam


def radec_to_xyz(ra, dec, time, location):
    """Convert ra and dec coordinates to ITRS coordinates for LOFAR observations.

    Args:
        ra (astropy Quantity): right ascension
        dec (astropy Quantity): declination
        time (float): MJD time in seconds
    Returns:
        pointing_xyz (ndarray): NumPy array containing the X, Y and Z coordinates
    """
    obstime = Time(time / 3600 / 24, scale="utc", format="mjd")
    loc_LOFAR = location

    # EarthLocation(
    #    lon=0.11990128407256424 * u.rad,
    #    lat=0.9203091252660295 * u.rad,
    #    height=6364618.852935438 * u.m,
    # )

    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_altaz = dir_pointing.transform_to(
        AltAz(obstime=obstime, location=loc_LOFAR)
    )
    dir_pointing_xyz = dir_pointing_altaz.transform_to(ITRS)

    pointing_xyz = np.asarray(
        [dir_pointing_xyz.x, dir_pointing_xyz.y, dir_pointing_xyz.z]
    )
    return pointing_xyz


station = "CS002LBA"
rcumode = "outer"
cs002 = mydb.phase_centres[station]

ref_pos = EarthLocation.from_geocentric(*cs002, unit=u.m)
latlonel = [ref_pos.lat.rad, ref_pos.lon.rad, ref_pos.height.value]

az = np.linspace(0, 355, 36)
el = np.linspace(30, 90, 6)
azel = AltAz(
    az=np.radians(az)[np.newaxis] * u.rad,
    alt=np.radians(el)[:, np.newaxis] * u.rad,
    location=ref_pos,
)


phasedir = SkyCoord.from_name("CAS A")
times = Time.now() + np.arange(10) * u.min

freqs = np.linspace(40e6, 60e6,3) * u.Hz

dynspec, distance_phase_center, distance_dir = getBeamPower(
    station, rcumode, azel.flatten(), phasedir, times, freqs
)
skypower =np.array(get_sky(times, freqs, latlonel)).swapaxes(0,1)  # freq x time x points
# make sure the beampattern (dynspec) has the same grid as skypower
beampower, new_az, new_zenith = interpolate_beam(dynspec.reshape(dynspec.shape[:-1] + azel.az.shape), 90-el, az)
beampower /= np.sum(beampower,axis=-1, keepdims=True)

noise_power = np.sum( beampower * skypower , axis=-1)