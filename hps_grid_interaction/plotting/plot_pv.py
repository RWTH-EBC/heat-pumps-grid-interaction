import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hps_grid_interaction.bes_simulation import weather


def plot_feed_in():
    csv_files_folder = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\01_BESMod_Simulationen\MonovalentHC_oldbuildings_HR\csv_files")
    grid_sum = []
    for file in os.listdir(csv_files_folder):
        grid_sum.append(pd.read_csv(csv_files_folder.joinpath(file), index_col=0))
    grid_sum_heat_supply = sum([d.loc[:, "heat_supply"] for d in grid_sum])
    grid_sum_household = sum([d.loc[:, "household"] for d in grid_sum]) - grid_sum_heat_supply
    grid_sum_household_pv_bat = sum([d.loc[:, "household+pv+battery"] for d in grid_sum])
    grid_sum_household_pv = sum([d.loc[:, "household+pv"] for d in grid_sum])
    grid_sum_pv = grid_sum_household + grid_sum_heat_supply - grid_sum_household_pv
    print(grid_sum_household_pv_bat.max(), grid_sum_heat_supply.max(), grid_sum_household_pv.max())
    from ebcpy import TimeSeriesData
    if "newbuildings" in csv_files_folder.parent.name:
        mat_name = "0_EFH2010_standard_SingleDwelling.mat"
    else:
        mat_name = "0_EFH1970_standard_SingleDwelling.mat"
    df = TimeSeriesData(csv_files_folder.parent.joinpath("SimulationResults", mat_name), variable_names=[
        "weaDat.weaBus.HGloHor", "weaDat.weaBus.HDirNor", "weaDat.weaBus.TDryBul", "weaDat.weaBus.HDifHor"
    ]).to_df().loc[86400*2:]

    days = grid_sum_heat_supply.index / 86400

    fig, ax = plt.subplots(7, 1, sharex=True)
    ax[0].plot(days, grid_sum_household_pv)
    ax[0].axhline(grid_sum_household_pv.min(), color="black")
    ax[0].set_ylabel("$P_\mathrm{el,Sum}$ in kW")

    ax[1].plot(days, grid_sum_household)
    ax[1].set_ylabel("$P_\mathrm{elHou}$ in kW")

    ax[2].plot(days, grid_sum_heat_supply)
    ax[2].set_ylabel("$P_\mathrm{el,Hea}$ in kW")

    ax[3].plot(days, grid_sum_pv)
    ax[3].axhline(grid_sum_pv.max(), color="black")
    ax[3].set_ylabel("$P_\mathrm{el,PV}$ in kW")

    ax[4].plot(days, df.loc[:, "weaDat.weaBus.TDryBul"].values)
    ax[4].set_ylabel("$T_\mathrm{Oda}$ in °C")
    ax[5].plot(days, df.loc[:, "weaDat.weaBus.HGloHor"].values)
    ax[5].set_ylabel("Global Radiation")
    ax[6].plot(days, df.loc[:, "weaDat.weaBus.HDirNor"].values)
    ax[6].set_ylabel("Direct Radiation")
    ax[-1].set_xlabel("Days in year")
    plt.show()

    print(grid_sum)


if __name__ == '__main__':
    plot_feed_in()
