import sys

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord, FK5
from astropy.time import Time
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

from getDynspecBeam import getDynspec, mydb


def model_flux(calibrator, frequency, sun_true=False):
    """
    Calculates the model matrix for flux calibration for a range of known calibrators:
    J0133-3629, 3C48, Fornax A, 3C 123, J0444+2809, 3C138, Pictor A, Taurus A, 3C147, 3C196, Hydra A, Virgo A,
    3C286, 3C295, Hercules A, 3C353, 3C380, Cygnus A, 3C444, Cassiopeia A

    Input: the calibrator name, frequency range, and time range
    Output: the calibration matrix (in Jy)
    Code adapted original from Peijin Zhang
    """
    parameters = []

    Cal_dict = {'J0133-3629': [1.0440, -0.662, -0.225],
                '3C48': [1.3253, -0.7553, -0.1914, 0.0498],
                'For A': [2.218, -0.661],
                'ForA': [2.218, -0.661],
                '3C123': [1.8017, -0.7884, -0.1035, -0.0248, 0.0090],
                'J0444-2809': [0.9710, -0.894, -0.118],
                '3C138': [1.0088, -0.4981, -0.155, -0.010, 0.022, ],
                'Pic A': [1.9380, -0.7470, -0.074],
                'Tau A': [2.9516, -0.217, -0.047, -0.067],
                'PicA': [1.9380, -0.7470, -0.074],
                'TauA': [2.9516, -0.217, -0.047, -0.067],
                '3C147': [1.4516, -0.6961, -0.201, 0.064, -0.046, 0.029],
                '3C196': [1.2872, -0.8530, -0.153, -0.0200, 0.0201],
                'Hyd A': [1.7795, -0.9176, -0.084, -0.0139, 0.030],
                'Vir A': [2.4466, -0.8116, -0.048],
                'HydA': [1.7795, -0.9176, -0.084, -0.0139, 0.030],
                'VirA': [2.4466, -0.8116, -0.048],
                '3C286': [1.2481, -0.4507, -0.1798, 0.0357],
                '3C295': [1.4701, -0.7658, -0.2780, -0.0347, 0.0399],
                'Her A': [1.8298, -1.0247, -0.0951],
                'HerA': [1.8298, -1.0247, -0.0951],
                '3C353': [1.8627, -0.6938, -0.100, -0.032],
                '3C380': [1.2320, -0.791, 0.095, 0.098, -0.18, -0.16],
                'Cyg A': [3.3498, -1.0022, -0.225, 0.023, 0.043],
                'CygA': [3.3498, -1.0022, -0.225, 0.023, 0.043],
                '3C444': [3.3498, -1.0022, -0.22, 0.023, 0.043],
                'Cas A': [3.3584, -0.7518, -0.035, -0.071],
                'CasA': [3.3584, -0.7518, -0.035, -0.071],
                'casa': [3.3584, -0.7518, -0.035, -0.071]
                }

    if calibrator in Cal_dict.keys():
        parameters = Cal_dict[calibrator]
    else:
        parameters = [1., 0.]
        #raise ValueError(calibrator, "is not in the calibrators list")

    flux_model = 0
    frequency = frequency * 0.001 # convert from MHz to GHz
    for j, p in enumerate(parameters):
        flux_model += p * numpy.log10(frequency) ** j
    flux_model = 10 ** flux_model  # because at first the flux is in log10

    if sun_true:
        return flux_model * 10 ** (-4)  # convert form Jy to sfu
    else:
        return flux_model

station = "LV614LBA"
rcumode = 3
LV614 = mydb.phase_centres[station]
ref_pos = EarthLocation.from_geocentric(*LV614, unit=u.m)

latlonel = [ref_pos.lat.rad, ref_pos.lon.rad, ref_pos.height.value]

src_casa =  SkyCoord.from_name("Cas A") #SkyCoord(ra="23h23m24.000s", dec="+58d48m54.00s", frame='icrs') # Cas A

# Frequency range
subband_min = 150
subband_max = 311
freqs_ = 0 + (200 / 1024) * np.linspace(subband_min, subband_max, subband_max - subband_min)
freqs = freqs_ * 1000000

phasedir = SkyCoord.from_name("3C295")
start = datetime.strptime("2025-01-02T15:00:16", "%Y-%m-%dT%H:%M:%S")
times = np.arange(0, 46800)
times = times * timedelta(seconds=1)
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

casa_flux = model_flux("casa", freqs_, sun_true=False)

fig, ax = plt.subplots()
dynspec_ = np.zeros(dynspec.shape)
for f in range(0,dynspec.shape[1]):
    dynspec_[:, f] = dynspec[:, f] * casa_flux

print("dynspec_dynspec_.shape", dynspec_.shape)
im1 = ax.imshow(dynspec_, aspect="auto", extent=[md.date2num(times[0]),md.date2num(times[-1]), freqs_[-1], freqs_[0]],
                vmin=np.percentile(dynspec_, 1), vmax=np.percentile(dynspec_, 99))

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
