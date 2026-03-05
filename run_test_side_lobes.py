import os
import sys

from lofarantpos.db import LofarAntennaDatabase
from tqdm import tqdm


def main():
    mydb = LofarAntennaDatabase()

    rcu_modes_ = {"LBA":[1, 2, 3, 4], "HBA":[5,6,7]}
    subband_min = 0
    subband_max = 511
    source = "3C295"
    start_time = "2025-01-02T15:00:16"
    duration = 46800

    processed_stations = []
    for station in tqdm(mydb.antennas):
        station_name  = station.station + station.antenna_type
        if station_name not in processed_stations:
            processed_stations.append(station_name)
        else:
            continue

        rcu_modes = rcu_modes_[station.antenna_type]
        for mode in rcu_modes:
            output_dir_name = "./" + station.station + "/" + station.antenna_type + "/rcu_mode"  + str(mode)  + "/"
            os.system("mkdir -p " + output_dir_name)

            # python3.10 test_side_lobes.py LV614LBA 3 150 311  3C295 2025-01-02T15:00:16 46800
            print("python3.10 test_side_lobes.py " + station_name + " " + str(mode) + " " + str(subband_min) +
                  " " + str(subband_max) + " " + source + " " + start_time + " " + str(duration)
                  + " --output_dir_name " + output_dir_name)
            os.system("python3.10 test_side_lobes.py " + station_name + " " + str(mode) + " " + str(subband_min) +
                      " " + str(subband_max) + " " + source + " " + start_time + " " + str(duration)
                      + " --output_dir_name " + output_dir_name)


if __name__ == "__main__":
    main()
    sys.exit(0)
