import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.monte_carlo.plots import _get_quota_value, _get_quota_name_except_value

metric_data = {
    "p_trafo": {"label": "$P$ in kVA", "opt": "max"},
    "q_trafo": {"label": "$Q$ in kVA", "opt": "max"},
    "s_trafo": {"label": "$S$ in kVA", "opt": "max"},
    "vm_pu_min": {"label": "$V_\mathrm{min}$ in p.u.", "opt": "min",
                  "axhlines": [0.9, 0.95, 0.97]},
    "max_line_loading": {"label": "$p_\mathrm{max}$ in kVA", "opt": "max"},
}


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


def load_case_and_trafo_data(path: Path):
    import json
    with open(path.joinpath("results_to_plot.json"), "r") as file:
        results = json.load(file)
    case_and_tafo_data = {}
    for grid, data in results.items():
        for case, case_data in data["grid"].items():
            trafo_data = {}
            startswith = f"results_{case}_{grid}_HR_3-ph_"  # TODO REVERT
            for file_name in os.listdir(path):
                if file_name.startswith(startswith) and file_name.endswith(".json"):
                    trafo_size = int(file_name.replace(startswith, "").replace(".json", ""))
                    with open(path.joinpath(file_name), "r") as file:
                        trafo_data[trafo_size] = json.load(file)
            case_and_tafo_data[_get_quota_value(case, True)] = trafo_data
    return case_and_tafo_data


def plot_time_series(
        case_trafo_data: dict,
        metric: str,
        save_path, fixed_case: str = None, fixed_trafo_size: int = None):
    if fixed_case is None and fixed_trafo_size is None:
        raise ValueError("One of both case or trafo_size must be fixed")
    if fixed_case is not None and fixed_trafo_size is not None:
        raise ValueError("Not both case or trafo_size can be fixed")
    if fixed_trafo_size is None:
        _data = {f"{key} kVA": np.array(value[metric]) for key, value in case_trafo_data[fixed_case].items()}
        fig_title = f"Case: {fixed_case}"
        save_name = f"case={fixed_case}"
    else:
        _data = {case: np.array(trafo_data[fixed_trafo_size][metric]) for case, trafo_data in case_trafo_data.items()}
        fig_title = f"Transformer: {fixed_trafo_size} kVA"
        save_name = f"trafo={fixed_trafo_size}"

    fig, ax = plt.subplots(1, 2, sharey=True)
    t_oda = load_outdoor_air_temperature()
    bins = np.linspace(t_oda.values[1:-1, 0].min(), t_oda.values[1:-1, 0].max(), num=30)
    categories = pd.cut(t_oda.values[1:-1, 0], bins, labels=False)
    for _label, time_series_data in _data.items():
        curve = []
        for bin_idx in range(len(bins)):
            bin_mask = (categories == bin_idx)
            if np.any(bin_mask):
                if metric_data[metric]["opt"] == "min":
                    curve.append(time_series_data[bin_mask].min())
                else:
                    curve.append(time_series_data[bin_mask].max())
            else:
                curve.append(np.NAN)

        ax[0].plot(np.arange(len(time_series_data)) / 4, np.sort(time_series_data)[::-1],
                   label=_label)
        ax[1].plot(bins, curve, label=_label, linestyle="-")
    axhlines = metric_data[metric].get("axhlines", None)
    if axhlines is not None:
        for hline in axhlines:
            for _ax in ax:
                _ax.axhline(hline, color="black")
    fig.suptitle(fig_title)
    ax[0].set_ylabel(metric_data[metric]["label"])
    ax[1].set_xlabel("$T_\mathrm{Oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"{metric}_{save_name}.png"))


def plot_all_metrics_and_trafos(path):
    case_and_trafo_data = load_case_and_trafo_data(path)
    cases = list(case_and_trafo_data.keys())
    trafo_sizes = list(case_and_trafo_data[cases[0]])
    save_path = path.joinpath("plots")
    os.makedirs(save_path, exist_ok=True)
    for metric in metric_data.keys():
        only_one_trafo_is_enough = metric in ["p_trafo", "s_trafo", "q_trafo"]
        #only_one_trafo_is_enough = False
        for trafo_size in trafo_sizes:
            plot_time_series(case_trafo_data=case_and_trafo_data,
                             save_path=save_path,
                             metric=metric, fixed_trafo_size=trafo_size)
            if only_one_trafo_is_enough:
                break
        for case in cases:
            plot_time_series(case_trafo_data=case_and_trafo_data,
                             save_path=save_path,
                             metric=metric, fixed_case=case)


if __name__ == '__main__':
    from hps_grid_interaction import RESULTS_MONTE_CARLO_FOLDER
    PlotConfig.load_default()
    PATH = RESULTS_MONTE_CARLO_FOLDER.joinpath("Altbau_av_hyb_with_pv_bat")
    plot_all_metrics_and_trafos(PATH)