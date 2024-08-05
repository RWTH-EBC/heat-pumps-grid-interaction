import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hps_grid_interaction import KERBER_NETZ_XLSX


def plot_voltage_drop(with_all_feeders: bool = False):
    df_grid = pd.read_excel(KERBER_NETZ_XLSX, sheet_name="Kerber Netz oldbuildings", index_col=0)

    RESULTS_MONTE_CARLO_FOLDER = pathlib.Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\03_monte_carlo")
    folder = RESULTS_MONTE_CARLO_FOLDER.joinpath("oldbuildings_hybrid_EMob_HP", "argmean")
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
        #ax[3].plot(index, feeder_sums[9], label="9", marker="^")
        ax[3].set_ylabel("$P_\mathrm{el}$ in kW")
        ax[3].legend(ncol=2)

    ax[-1].set_xlabel("Time in d")
    plt.show()


if __name__ == '__main__':
    plot_voltage_drop(with_all_feeders=True)
