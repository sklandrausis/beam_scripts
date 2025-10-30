from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord, FK5
from astropy.time import Time

import astropy.units as u
import numpy as np
from getDynspecBeam import getBeamPower, mydb
from skycal import get_sky, interpolate_beam

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py
import sys


#based on code by Maaijke Mevius orginal code https://github.com/maaijke/beam_scripts

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


station = "LV614LBA"
rcumode = 3
LV614 = mydb.phase_centres[station]

ref_pos = EarthLocation.from_geocentric(*LV614, unit=u.m)
latlonel = [ref_pos.lat.rad, ref_pos.lon.rad, ref_pos.height.value]

az = np.linspace(0, 355, 36)
el = np.linspace(30, 90, 6)
azel = AltAz(
    az=np.radians(az)[np.newaxis] * u.rad,
    alt=np.radians(el)[:, np.newaxis] * u.rad,
    location=ref_pos,
)

phasedir = SkyCoord.from_name("3C295")
times = Time("2025-01-02T14:59:16") + np.arange(800) * u.min

times = times[::30]
print(times[0], times[1], times[-1], len(times))

coordinates_of_lofar = EarthLocation(x=3183318.032280000 * u.m, y=1276777.654760000*u.m, z=5359435.077 * u.m)

ra = "05h42m36.13789710s"
dec = "+49d51m07.2337139s"

source_ = SkyCoord(ra=ra, dec=dec, frame=FK5, equinox='J2000.0')
frame = AltAz(obstime=times, location=coordinates_of_lofar)
elevation_azimuth = source_.transform_to(frame)
elevation = elevation_azimuth.alt

subband_min = 150
subband_max = 311
freqs = 0 + (200 / 1024) * np.linspace(subband_min, subband_max, subband_max - subband_min)
freqs = freqs * u.MHz
freqs = freqs.to("Hz")

dynspec, distance_phase_center, distance_dir = getBeamPower(
    station, rcumode, azel.flatten(), phasedir, times, freqs
)


beampower, new_az, new_zenith = interpolate_beam(dynspec.reshape(dynspec.shape[:-1] + azel.az.shape), 90-el, az)
beampower /= np.sum(beampower,axis=-1, keepdims=True)

skypower = np.array(get_sky(times, freqs, latlonel)).swapaxes(0,1)  # freq x time x points


noise_power = np.sum(beampower * skypower , axis=-1)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
index = 0
for time in times:
    ax1.plot(freqs, noise_power[:, index], label=str(time))
    index += 1

ax1.set_xlabel("Frequencies [Hz]")
ax1.set_ylabel("Power [K]")
ax1.legend()

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
im1 = ax2.imshow(np.flip(noise_power.T, axis=0), aspect="auto", extent=[freqs[0].value, freqs[-1].value,
                                                                  elevation[-1].value, elevation[0].value])
ax2.set_xlabel("Frequencies [Hz]")
ax2.set_ylabel("Elevation [Deg]")

divider = make_axes_locatable(ax2)
cax1 = divider.append_axes("right", size="5%", pad=0.07, label="K")
plt.colorbar(im1, ax=ax2, cax=cax1, label="K")

plt.show()

noise_power_output = h5py.File("/mnt/LOFAR0/noise_power.h5", "w")
noise_power_output.create_dataset("elevation", data=elevation.value)
noise_power_output.create_dataset("Frequencies", data=freqs.value)
noise_power_output.create_dataset("noise_power", data=noise_power)
noise_power_output.close()

sys.exit(0)
