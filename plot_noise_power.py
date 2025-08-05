import argparse
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main(noise_power_file):
    noise_power_file_data = h5py.File(noise_power_file, "r")
    elevation = noise_power_file_data["elevation"][()]
    frequencies = noise_power_file_data["Frequencies"][()]
    noise_power = noise_power_file_data["noise_power"][()]

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    ax1.plot(frequencies, noise_power)

    ax1.set_xlabel("Frequencies [Hz]")
    ax1.set_ylabel("Power [K]")
    ax1.legend()

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), dpi=150)
    im1 = ax2.imshow(np.flip(noise_power.T, axis=0), aspect="auto", extent=[frequencies[0], frequencies[-1],
                                                                            elevation[-1], elevation[0]])
    ax2.set_xlabel("Frequencies [Hz]")
    ax2.set_ylabel("Elevation [Deg]")

    divider = make_axes_locatable(ax2)
    cax1 = divider.append_axes("right", size="5%", pad=0.07, label="K")
    plt.colorbar(im1, ax=ax2, cax=cax1, label="K")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot noise power')
    parser.add_argument('noise_power_file', type=str, help='file name of noise_power_file')
    args = parser.parse_args()
    main(args.noise_power_file)
    sys.exit(0)
