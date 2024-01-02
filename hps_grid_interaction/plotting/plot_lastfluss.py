import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import load_outdoor_air_temperature


def extract_data(path: Path):
    df = pd.read_excel(path, index_col=0)
    df = df.dropna(axis=1)
    df.loc[:, "min"] = df.min(axis=1)
    df.loc[:, "max"] = df.max(axis=1)
    df.loc[:, "mean"] = df.mean(axis=1)
    df = df.loc[:, ["min", "max", "mean"]]
    out_path = path.parent.joinpath(path.stem + "_extracted.xlsx")
    df.to_excel(out_path)
    return out_path


def get_percent_smaller_than(df, metric, threshold):
    return np.count_nonzero(df.loc[:, metric] < threshold) / len(df) * 100


def analysis(base_path: Path):
    case_data = {}

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for case_name in os.listdir(base_path):
        if case_name.endswith(".xlsx") or is_float(case_name):
            continue
        print("Loading case", case_name)
        file_path = base_path.joinpath(case_name, "res_bus", "vm_pu_extracted.xlsx")
        if not file_path.exists():
            file_path = extract_data(base_path.joinpath(case_name, "res_bus", "vm_pu.xlsx"))
        plot_voltage(file_path)
        df_v = pd.read_excel(file_path, index_col=0)
        file_path = base_path.joinpath(case_name, "res_line", "loading_percent_extracted.xlsx")
        if not file_path.exists():
            file_path = extract_data(base_path.joinpath(case_name, "res_line", "loading_percent.xlsx"))
        df_line = pd.read_excel(file_path, index_col=0)
        case_data[case_name] = {
            "min_V_pu": df_v.loc[:, "min"].min(),
            "max_line": df_line.loc[:, "max"].max(),
            "time_smaller_95": get_percent_smaller_than(df_v, "min", 0.95),
            "time_smaller_97": get_percent_smaller_than(df_v, "min", 0.97)
        }
    pd.DataFrame(case_data).to_excel(base_path.joinpath("analysis.xlsx"))


def plot_voltage(path):
    df = pd.read_excel(path, index_col=0)
    df.index /= 4
    df_oda = load_outdoor_air_temperature()
    df_oda = df_oda.loc[:df.index[-1]]
    for variable in ["min", "max", "mean"]:
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[1].set_xlabel("$T_\mathrm{oda}$ in °C")
        ax[0].set_xlabel("Time in h")
        ax[0].set_ylabel(f"{variable.casefold()} Voltage in p.u.")
        ax[0].plot(df.index, df.loc[:, variable])
        ax[1].scatter(df_oda, df.loc[:, variable], s=1)
        ax[1].set_ylim([0.94, 1])
        ax[0].set_ylim([0.94, 1])
        fig.tight_layout()
        fig.savefig(path.parent.joinpath(path.stem + f"_{variable}.png"))
    plt.close(fig)
    #plt.show()


if __name__ == '__main__':
    PLOT_CONFIG = PlotConfig.parse_json_file(Path(__file__).parents[2].joinpath("data", "default_configs", "plotting.json"))
    PATH = Path(r"D:\01_Projekte\09_HybridWP\01_Results\03_lastfluss\LastflussSimulationenGEGBiv-RONT\3-ph")
    analysis(base_path=PATH)
