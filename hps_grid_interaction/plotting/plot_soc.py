import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ebcpy import TimeSeriesData
from hps_grid_interaction.bes_simulation import weather


def plot_soc():
    csv_files_folder = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\01_BESMod_Simulationen\MonovalentWeather_oldbuildings\SimulationResults")
    grid_sum = []
    for file in os.listdir(csv_files_folder):
        if not file.endswith(".hdf"):
            continue
        df = TimeSeriesData(csv_files_folder.joinpath(file)).to_df()
        grid_sum.append(df)

    sum_soc = sum([d.loc[:, "outputs.electrical.dis.SOCBat"] for d in grid_sum]) / len(grid_sum) * 100
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.scatter(grid_sum[-1].loc[:, "weaDat.weaBus.TDryBul"] - 273.15, sum_soc, color="red")
    ax.set_ylabel("Average $SoC$ in %")
    ax.axhline(20, label="Minimum $SoC$", color="black")
    ax.legend()
    ax.set_xlabel("$T_\mathrm{Oda}$ in °C")
    plt.show()


if __name__ == '__main__':
    plot_soc()
