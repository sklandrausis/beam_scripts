import argparse
import os
import sys

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

from getDynspecBeam import getDynspec, mydb

plt.style.use(os.path.expanduser('~') + "/.config/lofar/plot.style")

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

def sb_to_freq(subband_min, subband_max, rcumode, clock):
    n=1

    if rcumode == 1 or rcumode == 2 or rcumode == 3 or rcumode == 4:  # 0 MHz - 100 MHz
        n = 1
    elif rcumode == 5:  # 100 MHz - 200 MHz
        n = 2
    else: # 200 MHz - 300 MHz
        n = 3

    return np.linspace((n-1 + (subband_min/512))*(clock/2), (n-1 + (subband_max/512))*(clock/2), subband_max - subband_min + 1) #MHz

def main(station, rcumode, subband_min,  subband_max,  target_source, start_time, duration, clock=200, output_dir_name="/mnt/LOFAR0/beam_scripts/"):
    station_coordinates = mydb.phase_centres[station]

    # Frequency range
    freqs_ = sb_to_freq(subband_min, subband_max, rcumode, clock)
    freqs = freqs_ * 1000000  # Convert MHz to Hz

    phasedir = SkyCoord.from_name(target_source)
    start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    times = np.arange(0, duration)
    times = times * timedelta(seconds=1)
    times = start + times

    '''
    For HydA and Hyd A, I got this error. In future coordinates should add in code.
    astropy.coordinates.name_resolve.NameResolveError: Unable to find coordinates for name 'HydA'
    '''

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    ax2.scatter(phasedir.ra, phasedir.dec, 100, label=target_source)
    target_source_flux = model_flux(target_source, freqs_, sun_true=False)
    a_team_sources = ["Cas A", "Cyg A", "Tau A", "For A", "Her A", "Pic A"]
    a_team_sum = np.zeros((len(freqs), len(times)))

    for a_team_source in a_team_sources:
        print("Processing A-Team source", a_team_source)

        a_team_source_sky_coords = SkyCoord.from_name(a_team_source)
        dynspec, distance_phase_center, distance_dir = getDynspec(station, rcumode, a_team_source_sky_coords, phasedir,
                                                                  times, freqs * u.Hz)
        ateam_source_flux = model_flux(a_team_source, freqs_, sun_true=False)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
        ax.set_title(a_team_source)
        dynspec_ = np.zeros(dynspec.shape)
        for f in range(0,dynspec.shape[1]):
            dynspec_[:, f] = dynspec[:, f] * (ateam_source_flux/target_source_flux)

        a_team_sum += dynspec_
        im1 = ax.imshow(dynspec_, aspect="auto", extent=[md.date2num(times[0]),md.date2num(times[-1]), freqs_[-1], freqs_[0]],
                        vmin=np.percentile(dynspec_, 1), vmax=np.percentile(dynspec_, 99))

        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.07, label="flux ratio")
        plt.colorbar(im1, ax=ax, cax=cax1, label="flux ratio")

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(md.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_ylabel("Frequencies [MHz]", fontweight='bold')
        ax.set_xlabel("Time", fontweight='bold')

        ax2.scatter(a_team_source_sky_coords.ra, a_team_source_sky_coords.dec, 100, label=a_team_source)

        ax2.set_xlabel("RA [deg]", fontweight='bold')
        ax2.set_ylabel("DEC [deg]", fontweight='bold')

        print("Separation [deg]", a_team_source_sky_coords.separation(phasedir).deg)

        np.save(output_dir_name +  a_team_source.replace(" ", ""), dynspec_)

    fig_a_team_sum, ax_a_team_sum = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    ax_a_team_sum.set_title("a team sum")
    im1_a_team_sum = ax_a_team_sum.imshow(a_team_sum, aspect="auto",
                    extent=[md.date2num(times[0]), md.date2num(times[-1]), freqs_[-1], freqs_[0]],
                    vmin=np.percentile(a_team_sum, 1), vmax=np.percentile(a_team_sum, 99))

    divider_ax_a_team_sum = make_axes_locatable(ax_a_team_sum)
    cax1_ax_a_team_sum = divider_ax_a_team_sum.append_axes("right", size="5%", pad=0.07, label="flux ratio")
    plt.colorbar(im1_a_team_sum, ax=ax_a_team_sum, cax=cax1_ax_a_team_sum, label="flux ratio")

    ax_a_team_sum.xaxis_date()
    ax_a_team_sum.xaxis.set_major_formatter(md.ConciseDateFormatter(ax_a_team_sum.xaxis.get_major_locator()))
    ax_a_team_sum.set_ylabel("Frequencies [MHz]", fontweight='bold')
    ax_a_team_sum.set_xlabel("Time", fontweight='bold')

    np.save(output_dir_name + "a_team_sum", a_team_sum)

    ax2.legend()
    plt.show()

if __name__ == "__main__":
    #start_time = "2025-01-02T15:00:16"
    #station = LV614LBA
    #duration = 46800

    parser = argparse.ArgumentParser(description='Create side lobes model for given target source')
    parser.add_argument('station', type=str, help='name of the station')
    parser.add_argument('rcumode', type=int, help='rcu mode of the observation')
    parser.add_argument('subband_min', type=int, help='smallest subband')
    parser.add_argument('subband_max', type=int, help='largest subband')
    parser.add_argument('target_source', type=str, help='Target source of the observation')
    parser.add_argument('start_time', type=str, help='start time of the observation in format '
                                                     '"%Y-%m-%dT%H:%M:%S"')
    parser.add_argument('duration', type=int, help='duration of the observation in seconds')
    parser.add_argument('-c', '--clock', type=int, help='station clock', default=200)
    parser.add_argument('-o', '--output_dir_name', type=str, help='output directory name',
                        default="/mnt/LOFAR0/beam_scripts/")

    args = parser.parse_args()

    main(args.station, args.rcumode, args.subband_min,  args.subband_max,  args.target_source,
         args.start_time, args.duration, args.clock, args.output_dir_name)
    sys.exit(0)
