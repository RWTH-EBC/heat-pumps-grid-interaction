import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hps_grid_interaction import KERBER_NETZ_XLSX, RESULTS_MONTE_CARLO_FOLDER
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size


def plot_voltage_drop(with_all_feeders: bool = False):
    df_grid = pd.read_excel(KERBER_NETZ_XLSX, sheet_name="Kerber Netz oldbuildings", index_col=0)

    folder = RESULTS_MONTE_CARLO_FOLDER.joinpath("oldbuildings_hybrid_EMob_HP", "percentile_997")
    case = "hybrid_quota_EMob_HP_0"
    csv_inputs = folder.joinpath(f"grid_simulation_{case}")
    json_file = folder.joinpath(f"results_{case}_oldbuildings_3-ph_1400.json")
    with open(json_file, "r") as file:
        data = json.load(file)
    rev = {v: k for k, v in data["vm_pu_min_per_line"].items()}
    print("line with minimal voltage", rev[min(rev)])
    feeder_sums = {}
    for csv_file in os.listdir(csv_inputs):
        if not csv_file.endswith(".csv"):
            continue
        idx = int(csv_file.replace(".csv", ""))
        feeder = int(df_grid.loc[idx, "Anschlusspunkt"].split("-")[0])
        df = pd.read_csv(csv_inputs.joinpath(csv_file), index_col=0)
        feeder_sums[feeder] = df.values + feeder_sums.get(feeder, 0)

    index = np.arange(len(data["vm_pu_min"])) / 4 / 24
    voltage = data["vm_pu_min"]
    power = data["p_trafo"]
    fig, ax = plt.subplots(4 if with_all_feeders else 3, 1, sharex=True)

    from hps_grid_interaction.utils import load_outdoor_air_temperature
    t_oda_csv = load_outdoor_air_temperature()
    t_oda_csv.index /= 24
    ax[0].plot(t_oda_csv.index, t_oda_csv)
    ax[0].set_ylabel("$T_\mathrm{Oda}$ in °C")

    ax[1].plot(index, voltage)
    ax[1].set_ylabel("$v_\mathrm{min}$ in p.u.")

    ax[2].plot(index, power)
    ax[2].plot(index, np.sum(list(feeder_sums.values()), axis=0), linestyle="--")
    ax[2].set_ylabel("$P_\mathrm{el,Ges}$ in kW")
    if with_all_feeders:
        for idx, feeder in enumerate(feeder_sums):
            ax[3].plot(index, feeder_sums[feeder], label=feeder)
        # ax[3].plot(index, feeder_sums[9], label="9", marker="^")
        ax[3].set_ylabel("$P_\mathrm{el}$ in kW")
        ax[3].legend(ncol=2)

    ax[-1].set_xlabel("Time in d")
    plt.show()


def plot_operation_at_voltage_drop(with_all_feeders: bool = False):
    df_grid = pd.read_excel(KERBER_NETZ_XLSX, sheet_name="Kerber Netz oldbuildings", index_col=0)
    fig, ax = plt.subplots(
        4 if with_all_feeders else 3, 1, sharex=True,
        figsize=get_figure_size(n_columns=1, height_factor=1.5))
    end = 29 * 24
    start = 28 * 24

    from hps_grid_interaction.utils import load_outdoor_air_temperature
    t_oda_csv = load_outdoor_air_temperature()
    mask_oda = (t_oda_csv.index > start) & (t_oda_csv.index < end)

    folder = RESULTS_MONTE_CARLO_FOLDER.joinpath("oldbuildings_hybrid_EMob_HP", "argmean")
    plt_kwargs = {
        "hybrid_quota_EMob_HP_0": {"color": EBCColors.blue, "linestyle": "-", "label": "0 %"},
        "hybrid_quota_EMob_HP_40": {"color": EBCColors.grey, "linestyle": "-", "label": "40 %"},
        "hybrid_quota_EMob_HP_60": {"color": EBCColors.green, "linestyle": "-", "label": "60 %"},
    }
    for case in plt_kwargs:
        csv_inputs = folder.joinpath(f"grid_simulation_{case}")
        json_file = folder.joinpath(f"results_{case}_oldbuildings_3-ph_1400.json")
        with open(json_file, "r") as file:
            data = json.load(file)
        rev = {v: k for k, v in data["vm_pu_min_per_line"].items()}
        print("line with minimal voltage", rev[min(rev)])
        feeder_sums = {}
        for csv_file in os.listdir(csv_inputs):
            if not csv_file.endswith(".csv"):
                continue
            idx = int(csv_file.replace(".csv", ""))
            feeder = int(df_grid.loc[idx, "Anschlusspunkt"].split("-")[0])
            df = pd.read_csv(csv_inputs.joinpath(csv_file), index_col=0)
            feeder_sums[feeder] = df.values + feeder_sums.get(feeder, 0)

        ax[-1].plot(t_oda_csv.index[mask_oda], t_oda_csv[mask_oda], color=EBCColors.red)
        ax[-1].set_ylabel("$T_\mathrm{Oda}$ in °C")

        index = np.arange(len(data["vm_pu_min"])) / 4
        mask = (index > start) & (index < end)
        index = index[mask]
        voltage = np.array(data["vm_pu_min"])[mask]
        power = np.array(data["p_trafo"])[mask]

        ax[0].plot(index, voltage, **plt_kwargs[case])
        ax[1].plot(index, power, **plt_kwargs[case])
        if with_all_feeders:
            for idx, feeder in enumerate(feeder_sums):
                ax[2].plot(index, feeder_sums[feeder][mask], label=feeder)
            # ax[3].plot(index, feeder_sums[9], label="9", marker="^")
            ax[2].set_ylabel("$P_\mathrm{el}$ in kW")
            ax[2].legend(ncol=2)
    ax[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.02 , 1, 0.02), mode="expand")
    ax[0].set_ylabel("$v_\mathrm{min}$ in p.u.")
    ax[1].set_ylabel("$P_\mathrm{el,Ges}$ in kW")
    ax[-1].set_xlabel("Time in h")
    fig.tight_layout()
    fig.savefig("voltage_drop_day.png")
    plt.show()


if __name__ == '__main__':
    plot_operation_at_voltage_drop(with_all_feeders=False)
