import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hps_grid_interaction import RESULTS_BES_FOLDER
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size
from hps_grid_interaction.plotting.config import PlotConfig
from ebcpy import TimeSeriesData


def plot_hybrid_operation(with_all_feeders: bool = False):
    boi = "outputs.hydraulic.dis.QBoi_flow.value"
    hp = "outputs.hydraulic.gen.QHeaPum_flow.value"
    TOda = "weaDat.weaBus.TDryBul"
    variable_names = [boi, hp, TOda]
    old = True
    if old:
        folder = RESULTS_BES_FOLDER.joinpath("HybridWeather_oldbuildings", "SimulationResults")
        min_case = TimeSeriesData(folder.joinpath("215_EFH1950_adv_retrofit_SingleDwelling.mat"),
                                  variable_names=variable_names)
        max_case = TimeSeriesData(folder.joinpath("429_MFH_10_WE_1970_standard_SingleDwelling.mat"),
                                  variable_names=variable_names)
    else:
        folder = RESULTS_BES_FOLDER.joinpath("HybridWeather_newbuildings", "SimulationResults")
        min_case = TimeSeriesData(folder.joinpath("2_EFH2010_standard_SingleDwelling.mat"),
                                  variable_names=variable_names)
        max_case = TimeSeriesData(folder.joinpath("236_MFH_10_WE_1980_adv_retrofit_SingleDwelling.mat"),
                                  variable_names=variable_names)

    cases = [min_case, max_case]
    fig, ax = plt.subplots(len(cases), 1, sharex=True,
                           figsize=get_figure_size(n_columns=1, height_factor=1.5))
    for i, case in enumerate(cases):
        case.index /= 86400
        case = case.loc[63:70]
        ax[i].plot(case.index, case.loc[:, boi] / 1000, color="red", label="Boiler")
        ax[i].plot(case.index, case.loc[:, hp] / 1000, color="blue", label="HP")
        ax[i].set_ylabel("$\dot{Q}$ in kW")
        mask_to_spans(ax[i], case.loc[:, TOda] < 274.15, case.index)
    ax[0].legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.02, 1, 0.02))
    ax[-1].set_xlabel("Time in d")
    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig("hybrid_opereation.png")
    plt.show()


def mask_to_spans(ax, mask, index, alpha=0.3, color='gray'):
    # Find transitions
    mask = mask.values[:, 0]
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]

    # Create spans
    for start, end in zip(starts, ends):
        end = min(len(index) - 1, end)
        start = min(len(index) - 1, start)
        ax.axvspan(index[start], index[end], color=color, alpha=alpha)


if __name__ == '__main__':
    plt.rcParams.update({"figure.figsize": [6.24 * 1.2, 5.78 / 1.3], "font.size": 11, "figure.dpi": 500})
    plot_hybrid_operation()
