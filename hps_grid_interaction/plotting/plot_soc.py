import os

import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData
from hps_grid_interaction import RESULTS_BES_FOLDER


def plot_soc():
    csv_files_folder = RESULTS_BES_FOLDER.joinpath("MonovalentWeather_oldbuildings", "\SimulationResults")
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
