import argparse
import os
import sys

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy

import numpy as np
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dreambeam.rime.scenarios import on_pointing_axis_tracking

from getDynspecBeam import getDynspec, mydb

plt.style.use(os.path.expanduser('~') + "/.config/lofar/plot.style")

def get_jones_gain(jones_xx, jones_yy, jones_xy, jones_yx):
    return np.sqrt(jones_xx**2 + jones_yy**2 + jones_xy**2 + jones_yx**2)


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

    fig_zenith_angle, ax_zenith_angle = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    fig_zenith_angle_cos, ax_zenith_angle_cos = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)

    station_coordinates = EarthLocation.from_geocentric(*mydb.phase_centres[station], unit=u.m)
    frame = AltAz(obstime=times, location=station_coordinates)

    elevation_azimuth_target_source = phasedir.transform_to(frame)
    elevation = elevation_azimuth_target_source.alt
    zenith_angle = 90 - elevation.value
    ax_zenith_angle.scatter(md.date2num(times), zenith_angle, label=target_source)
    ax_zenith_angle_cos.scatter(md.date2num(times), np.cos(np.deg2rad(zenith_angle)) **2, label=target_source)

    obstimestp = timedelta(seconds=1)
    obstimebeg = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    pointingdir = (np.deg2rad(phasedir.ra.deg), np.deg2rad(phasedir.dec.deg), 'J2000')
    samptimes, freqs_joins, jones, jonesobj = on_pointing_axis_tracking('LOFAR', "LV614",
                                                                  'LBA', "Hamaker", obstimebeg,
                                                                  timedelta(seconds=duration-1), obstimestp, pointingdir)

    freqs_joins_index_min = freqs_joins.index(freqs[0])
    freqs_joins_index_max = freqs_joins.index(freqs[-1]) + 1
    jones_inv = np.linalg.inv(jones)

    jones_inv_select_freq_3c = jones_inv[freqs_joins_index_min:freqs_joins_index_max, :]
    jones_xx_target = jones_inv_select_freq_3c[:, :, 0, 0]
    jones_yy_target = jones_inv_select_freq_3c[:, :, 1, 1]
    jones_xy_target = jones_inv_select_freq_3c[:, :, 1, 0]
    jones_yx_target = jones_inv_select_freq_3c[:, :, 0, 1]

    jones_inv_select_freq_3c[np.isnan(jones_inv_select_freq_3c)] = 0

    del samptimes, freqs_joins, jones, jonesobj

    print("JONES XX max, min for target source ", np.max(jones_xx_target), np.min(jones_xx_target))

    jones_i_target = (jones_xx_target + jones_yy_target)
    #jones_gain_target = get_jones_gain(jones_xx_target, jones_yy_target, jones_xy_target, jones_yx_target)

    fig_jones_i, ax_jones_i = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    ax_jones_i.set_title("jones " + target_source)
    im1_jones_i = ax_jones_i.imshow(np.abs(jones_i_target), aspect="auto",
                                          extent=[md.date2num(times[0]), md.date2num(times[-1]), freqs_[-1], freqs_[0]])

    divider_jones_i = make_axes_locatable(ax_jones_i)
    cax1_ax_jones_i = divider_jones_i.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(im1_jones_i, ax=ax_jones_i, cax=cax1_ax_jones_i)

    ax_jones_i.xaxis_date()
    ax_jones_i.xaxis.set_major_formatter(md.ConciseDateFormatter(ax_jones_i.xaxis.get_major_locator()))
    ax_jones_i.set_ylabel("Frequencies [MHz]", fontweight='bold')
    ax_jones_i.set_xlabel("Time", fontweight='bold')

    print("\n\n\n")
    for a_team_source in a_team_sources:
        print("Processing A-Team source", a_team_source)
        a_team_source_sky_coords = SkyCoord.from_name(a_team_source)

        pointingdir = (np.deg2rad(a_team_source_sky_coords.ra.deg), np.deg2rad(a_team_source_sky_coords.dec.deg), 'J2000')
        samptimes, freqs_joins, jones, jonesobj = on_pointing_axis_tracking('LOFAR', "LV614",
                                                                            'LBA', "Hamaker", obstimebeg,
                                                                            timedelta(seconds=duration - 1), obstimestp,
                                                                            pointingdir)

        try:
            jones_inv = np.linalg.inv(jones)

            jones_inv_select_freq_ateam = jones_inv[freqs_joins_index_min:freqs_joins_index_max, :]
            jones_xx_ateam = jones_inv_select_freq_ateam[:, :, 0, 0]
            jones_yy_ateam = jones_inv_select_freq_ateam[:, :, 1, 1]
            jones_xy_ateam = jones_inv_select_freq_ateam[:, :, 1, 0]
            jones_yx_ateam = jones_inv_select_freq_ateam[:, :, 0, 1]

            del samptimes, freqs_joins, jones, jonesobj

            print("JONES xx max, min for A-Team source " + a_team_source, np.max(np.abs(jones_xx_ateam)), np.min(np.abs(jones_xx_ateam)))

            if np.sum(np.abs(jones_xx_ateam)) != 0:

                #jones_i_ateam = (jones_xx_ateam + jones_yy_ateam) / 2
                jones_i_ateam = jones_xx_ateam + jones_yy_ateam #get_jones_gain(jones_xx_ateam, jones_yy_ateam,  jones_xy_ateam, jones_yx_ateam)

                fig_jones_i, ax_jones_i = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
                ax_jones_i.set_title("jones " + a_team_source)
                im1_jones_i = ax_jones_i.imshow(np.abs(jones_i_ateam), aspect="auto",
                                                extent=[md.date2num(times[0]), md.date2num(times[-1]), freqs_[-1], freqs_[0]])

                divider_jones_i = make_axes_locatable(ax_jones_i)
                cax1_ax_jones_i = divider_jones_i.append_axes("right", size="5%", pad=0.07)
                plt.colorbar(im1_jones_i, ax=ax_jones_i, cax=cax1_ax_jones_i)

                ax_jones_i.xaxis_date()
                ax_jones_i.xaxis.set_major_formatter(md.ConciseDateFormatter(ax_jones_i.xaxis.get_major_locator()))
                ax_jones_i.set_ylabel("Frequencies [MHz]", fontweight='bold')
                ax_jones_i.set_xlabel("Time", fontweight='bold')

                dynspec, distance_phase_center, distance_dir = getDynspec(station, rcumode, a_team_source_sky_coords, phasedir,
                                                                          Time(times), freqs * u.Hz)


                np.save(output_dir_name + a_team_source.replace(" ", "") + "before_correction", dynspec)

                dynspec_flux = np.copy(dynspec)

                print("SIDE LOBES model max, min for A-Team source " + a_team_source, np.max(dynspec), np.min(dynspec))

                geometric_beam_matrix_multiplyde_by_inv_jones_ = np.zeros(jones_inv_select_freq_ateam.shape)

                geometric_beam_matrix_multiplyde_by_inv_jones_[:, :, 0, 0] = dynspec
                geometric_beam_matrix_multiplyde_by_inv_jones_[:, :, 1, 1] = dynspec

                jones_inv_select_freq_ateam[np.isnan(jones_inv_select_freq_ateam)] = 0

                jones_ratio = (jones_inv_select_freq_ateam / jones_inv_select_freq_3c)
                print("jones_ratio model max, min for A-Team source " + a_team_source, np.max(jones_ratio), np.min(jones_ratio))

                geometric_beam_matrix_multiplyde_by_inv_jones = np.matmul(geometric_beam_matrix_multiplyde_by_inv_jones_, jones_ratio)

                dynspec_corrected_by_pointing_jones = (geometric_beam_matrix_multiplyde_by_inv_jones[:, :, 0, 0]
                                                       + geometric_beam_matrix_multiplyde_by_inv_jones[:, :, 1, 1])

                dynspec_corrected_by_pointing_jones = np.abs(dynspec_corrected_by_pointing_jones)

                print("corrected beam model max, min for A-Team source " + a_team_source, np.max(dynspec), np.min(dynspec))

                ateam_source_flux = model_flux(a_team_source, freqs_, sun_true=False)

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
                ax.set_title(a_team_source)
                dynspec_ = np.zeros(dynspec.shape)

                #test (162, 162) (162, 46800) (162, 46800)
                print("test", dynspec_corrected_by_pointing_jones.shape, dynspec.shape, jones_ratio.shape)
                #sys.exit()

                for f in range(0,dynspec.shape[1]):
                    dynspec_[:, f] = dynspec_corrected_by_pointing_jones[:, f] * (ateam_source_flux/target_source_flux)
                    dynspec_flux[:, f] = dynspec_flux[:, f] * (ateam_source_flux/target_source_flux)

                np.save(output_dir_name + a_team_source.replace(" ", "") + "before_correction_flux", dynspec_flux)
                del dynspec_flux

                print("corrected beam model FLUX ratio max, min for A-Team source " + a_team_source, np.max(dynspec_),
                      np.min(dynspec_))

                dynspec_[np.isnan(dynspec_)] = 0
                dynspec_[np.isinf(dynspec_)] = 0

                print("corrected beam model FLUX normalized ratio max, min for A-Team source " + a_team_source, np.max(dynspec_),
                      np.min(dynspec_))

                a_team_sum += dynspec_

                print("new a_team_sum max, min for A-Team source " + a_team_source,
                      np.max(a_team_sum),
                      np.min(a_team_sum))

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
                np.save(output_dir_name + a_team_source.replace(" ", ""), dynspec_)

                del dynspec_

                elevation_azimuth = a_team_source_sky_coords.transform_to(frame)
                elevation = elevation_azimuth.alt
                zenith_angle = 90 - elevation.value
                ax_zenith_angle.scatter(md.date2num(times), zenith_angle, label=a_team_source)
                ax_zenith_angle_cos.scatter(md.date2num(times), np.cos(np.deg2rad(zenith_angle)) ** 2, label=a_team_source)

            else:
                print("new a_team_sum max, min for A-Team source " + a_team_source,
                      np.max(a_team_sum),
                      np.min(a_team_sum))

            #break
            print("\n\n\n")
            
        except:
            pass

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

    ax_zenith_angle.set_ylabel("Zenith angle [deg]")
    ax_zenith_angle.set_xlabel("Time")
    ax_zenith_angle.xaxis_date()
    ax_zenith_angle.xaxis.set_major_formatter(md.ConciseDateFormatter(ax_zenith_angle.xaxis.get_major_locator()))

    ax_zenith_angle_cos.set_ylabel(r'$cos (Zenith\ angle) ^2$')
    ax_zenith_angle_cos.set_xlabel("Time")
    ax_zenith_angle_cos.xaxis_date()
    ax_zenith_angle_cos.xaxis.set_major_formatter(md.ConciseDateFormatter(ax_zenith_angle_cos.xaxis.get_major_locator()))

    ax2.legend()
    ax_zenith_angle.legend()
    ax_zenith_angle_cos.legend()
    plt.show()

if __name__ == "__main__":
    # python3.10 test_side_lobes.py LV614LBA 3 150 311  3C295 2025-01-02T15:00:16 46800
    # python3.12 test_side_lobes.py LV614LBA 3 150 311  3C295 2025-01-02T15:00:16 100

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
