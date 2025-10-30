from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as md

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord, FK5
from astropy.time import Time
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

from getDynspecBeam import getDynspec, mydb

obs_time = Time('2025-01-02T14:59:16')
station = "LV614LBA"
rcumode = 3
LV614 = mydb.phase_centres[station]
ref_pos = EarthLocation.from_geocentric(*LV614, unit=u.m)

latlonel = [ref_pos.lat.rad, ref_pos.lon.rad, ref_pos.height.value]

src_casa =  SkyCoord.from_name("Cas A") #SkyCoord(ra="23h23m24.000s", dec="+58d48m54.00s", frame='icrs') # Cas A

# Frequency range
subband_min = 150
subband_max = 311
freqs = 0 + (200 / 1024) * np.linspace(subband_min, subband_max, subband_max - subband_min)

phasedir = SkyCoord.from_name("3C295")

start = datetime.strptime("2025-01-02T14:59:16", "%Y-%m-%dT%H:%M:%S")
times = np.arange(0, 800)
times = times * timedelta(minutes=1)
times = start + times

LV614 = mydb.phase_centres[station]

az = np.linspace(0, 355, 36)
el = np.linspace(30, 90, 6)
azel = AltAz(
    az=np.radians(az)[np.newaxis] * u.rad,
    alt=np.radians(el)[:, np.newaxis] * u.rad,
    location=ref_pos,
)

dynspec, distance_phase_center, distance_dir = getDynspec(station, rcumode, src_casa, phasedir, times, freqs)
print("dynspec.shape", dynspec.shape, len(times))
print("distance_phase_center.shape", distance_phase_center.shape, len(times))
print("distance_dir.shape", distance_dir.shape, len(times))

fig, ax = plt.subplots()
im1 = ax.imshow(dynspec, aspect="auto", extent=[md.date2num(times[0]),md.date2num(times[-1]), freqs[-1], freqs[0]],
                vmin=np.percentile(dynspec, 1), vmax=np.percentile(dynspec, 99))

divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.07,)
plt.colorbar(im1, ax=ax, cax=cax1)

ax.xaxis_date()
ax.xaxis.set_major_formatter(md.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_ylabel("Frequencies [MHz]", fontweight='bold')
ax.set_xlabel("Time", fontweight='bold')

fig2, ax2 = plt.subplots()
ax2.scatter(src_casa.ra, src_casa.dec, 100, label="Cas A")
ax2.scatter(phasedir.ra, phasedir.dec, 100, label="3C295")

ax2.set_xlabel("RA [deg]", fontweight='bold')
ax2.set_ylabel("DEC [deg]", fontweight='bold')
ax2.legend()

print("Separation [deg]", src_casa.separation(phasedir).deg)

plt.show()
