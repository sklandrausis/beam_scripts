import numpy as np
import matplotlib.pyplot as plt
import everybeam
import casacore.tables as pt
from astropy.coordinates import SkyCoord, ITRS
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time
from lofarantpos import db

mydb = db.LofarAntennaDatabase()
light_speed = 299792458.0


def prepare_ms(msname, newmsname, new_dir):
    """Copy any MS to a mock MS
    Args:
        msname (str): input msname
        newmsname (str): output msname
        new_dir (astropy.SkyCoord): new phase center
    """
    pt.taql(
        f"SELECT from {msname} where TIME in (SELECT TIME FROM :: LIMIT 1) giving {newmsname} as plain"
    )
    ra = new_dir.ra.deg
    dec = new_dir.dec.deg
    pt.taql(f"UPDATE {newmsname}::FIELD SET DELAY_DIR=[{ra,dec}]deg")
    pt.taql(f"UPDATE {newmsname}::FIELD SET REFERENCE_DIR=[{ra,dec}]deg")
    pt.taql(f"UPDATE {newmsname}::FIELD SET PHASE_DIR=[{ra,dec}]deg")
    pt.taql(f"UPDATE {newmsname}::FIELD SET LOFAR_TILE_BEAM_DIR=[{ra,dec}]deg")


def init_beam(fname):

    # o.use_differential_beam=use_differential_beam
    b = everybeam.load_telescope(fname)
    times = pt.taql("SELECT TIME from $fname orderby unique TIME").getcol("TIME")
    phase_dir = pt.table(fname + "/FIELD").getcol("PHASE_DIR")[0][0]
    return b, times, np.degrees(phase_dir)


def get_station_response(
    mybeam, direction, times, freqs, refdir, station_idx=0, rotate=True
):
    data = np.zeros(times.shape + freqs.shape + (2, 2), dtype=np.complex)
    atimes = Time(times / (3600 * 24.0), format="mjd")
    alldirs = get_itrf_dirs(atimes, direction, mybeam.station_name(station_idx))
    station0 = get_itrf_dirs(atimes, refdir, mybeam.station_name(station_idx))
    print(alldirs)
    print(station0)

    for itime, time in enumerate(times):
        for ifreq, freq in enumerate(freqs):

            data[itime, ifreq] = mybeam.station_response(
                time=time,
                freq=freq,
                station_idx=station_idx,
                direction=alldirs[itime],
                station0_direction=station0[itime],
                rotate=rotate,
            )
    return data


def get_itrf_dirs(times, direction, stationname="CS002LBA"):
    statpos = EarthLocation.from_geocentric(
        *list(mydb.phase_centres[stationname]), unit=u.m
    )
    # srcpos = SkyCoord(FK5(direction[0]*u.deg,direction[1]*u.deg),obstime = times)
    srcpos = SkyCoord(ra=direction[0] * u.deg, dec=direction[1] * u.deg, obstime=times)
    itrf_dirs = srcpos.transform_to(ITRS)
    return itrf_dirs.cartesian.xyz.value.T


def getbeamval(fname, srcdirs, freqs, station="CS002HBA1"):
    mybeam, times, refdir = init_beam(fname, True)
    st_idx = [mybeam.station_name(i) for i in range(mybeam.nr_stations)].index(station)
    beamval = np.zeros(
        (len(srcdirs),) + times.shape + freqs.shape + (2, 2), dtype=np.complex
    )
    for isrc, src in enumerate(srcdirs):
        beamval[isrc] = get_station_response(
            mybeam, src, times, freqs=freqs, refdir=refdir, station_idx=st_idx
        )
    return beamval


def get_clusterpos(clusterfile, srcfile, clusteridx=206):
    myf = open(clusterfile, "r")
    srcs = open(srcfile, "r")
    for line in myf:
        if line.strip().split()[0] == str(clusteridx):
            srclist = line.strip().split()[1:]
            break
    ras = []
    decs = []
    fluxs = []
    for src in srclist:
        srcs.seek(0)
        for line in srcs:
            if line.strip().split()[0] == src:
                data = line.strip().split()[1:]
                ra = (
                    (float(data[0]) + float(data[1]) / 60.0 + float(data[2]) / 3600.0)
                    * 180.0
                    / 12.0
                )
                dec = float(data[3]) + float(data[4]) / 60.0 + float(data[5]) / 3600.0
                flux = float(data[6])
                ras.append(ra)
                decs.append(dec)
                fluxs.append(flux)
                # print(data,ra,dec,flux)
    ras = np.array(ras)
    decs = np.array(decs)
    fluxs = np.array(fluxs)
    if np.abs(np.remainder(ras[np.argmax(fluxs)], 360)) < 20:
        ras = np.remainder(ras + 180.0, 360.0) - 180
    else:
        ras = np.remainder(ras, 360.0)

    ra = np.sum(ras * fluxs) / np.sum(fluxs)
    dec = np.sum(decs * fluxs) / np.sum(fluxs)

    return ra, dec, np.sum(fluxs)


def plotbeam(
    fname,
    station,
    clidx,
    clusterfile="sky_sagecal_no_diffuse_high_res_Ateam.txt.cluster",
    srcfile="sky_sagecal_no_diffuse.txt",
):
    ra, dec, _ = get_clusterpos(clusterfile, srcfile, clusteridx=clidx)
    freqs = np.arange(121.422e6, 160e6, 0.1953125e6)
    beam = getbeamval(fname, [[ra, dec]], freqs, station=station)
    return freqs, beam
