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


def get_percent_smaller_than(np_array, threshold):
    return np.count_nonzero(np_array < threshold) / len(np_array) * 100


def load_case_and_trafo_data(path: Path):
    import json
    with open(path.joinpath("results_to_plot.json"), "r") as file:
        results = json.load(file)
    case_and_tafo_data = {}
    for grid, data in results.items():
        for case, case_data in data["grid"].items():
            trafo_data = {}
            startswith = f"results_{case}_{grid}_3-ph_"  # TODO REVERT
            for file_name in os.listdir(path):
                if file_name.startswith(startswith) and file_name.endswith(".json"):
                    trafo_size = int(file_name.replace(startswith, "").replace(".json", ""))
                    with open(path.joinpath(file_name), "r") as file:
                        trafo_results = json.load(file)
                    trafo_data[trafo_size] = {metric: np.array(values) for metric, values in trafo_results.items()}
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
        _data = {f"{key} kVA": value[metric] for key, value in case_trafo_data[fixed_case].items()}
        fig_title = f"Case: {fixed_case}"
        save_name = f"case={fixed_case}"
    else:
        _data = {case: trafo_data[fixed_trafo_size][metric] for case, trafo_data in case_trafo_data.items()}
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
    plt.close("all")


def get_statistics(case_and_trafo_data: dict, path: Path):
    df = pd.DataFrame()
    idx = 0
    for case, trafo_data in case_and_trafo_data.items():
        for trafo_size, trafo_results in trafo_data.items():
            df.loc[idx, "Trafo-Size"] = trafo_size
            df.loc[idx, "case"] = case
            for metric, settings in metric_data.items():
                opt = settings["opt"]
                if opt == "min":
                    df.loc[idx, f"{metric}_{opt}"] = trafo_results[metric].min()
                if opt == "max":
                    df.loc[idx, f"{metric}_{opt}"] = trafo_results[metric].max()
                if "axhlines" in settings:
                    for axhline_value in settings["axhlines"]:
                        df.loc[idx, f"{metric} smaller {axhline_value}"] = get_percent_smaller_than(
                            trafo_results[metric], axhline_value
                        )
            idx += 1
    df.to_excel(path.joinpath(f"grid_statistics_{path.name}.xlsx"), sheet_name=path.name)
    return df


def plot_all_metrics_and_trafos(case_and_trafo_data: dict, path: Path):
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


def generate_all_cases(path: Path, with_plot: bool, altbau=True):
    if altbau:
        grid_case = "Altbau_"
    else:
        grid_case = "Neubau_"
    cases = [
        "av_e_mob_with_pv_bat",
        "av_heat_pump",
        "av_heating_rod",
        "av_pv_bat",
        "av_hyb",
        "av_pv",
        "show_extremas",
        "av_hyb_with_pv_bat",
    ]
    dfs = []
    for case in cases:
        case_path = PATH.joinpath(grid_case + case)
        case_and_trafo_data = load_case_and_trafo_data(case_path)
        if with_plot:
            plot_all_metrics_and_trafos(case_and_trafo_data, case_path)
        df = get_statistics(case_and_trafo_data, case_path)
        df.loc[:, "quota_cases"] = case
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.index = range(len(dfs))
    dfs.to_excel(path.joinpath(f"{grid_case}all_grid_results_ex.xlsx"))


if __name__ == '__main__':
    PlotConfig.load_default()
    PATH = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\03_monte_carlo")
    generate_all_cases(PATH, with_plot=True)
