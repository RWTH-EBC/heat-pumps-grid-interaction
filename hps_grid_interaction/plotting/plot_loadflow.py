import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.monte_carlo.monte_carlo import QuotaVariation
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size
from hps_grid_interaction.monte_carlo.plots import plot_quota_case_with_images
from hps_grid_interaction import DATA_PATH

metric_data = {
    "p_trafo": {"label": "$P$ in kW", "opt": "max"},
    "q_trafo": {"label": "$Q$ in kW", "opt": "max"},
    "s_trafo": {"label": "$S$ in kVA", "opt": "max"},
    "vm_pu_min": {"label": "$V_\mathrm{min}$ in p.u.", "opt": "min",
                  "axhlines": [0.9, 0.95, 0.97], "min_max": {"vmin": 0.9, "vmax": 1}},
#    "vm_pu_max": {"label": "$V_\mathrm{max}$ in p.u.", "opt": "max", "min_max": {"vmin": 0.85, "vmax": 1}},
    "max_line_loading": {"label": "$p_\mathrm{max}$ in %", "opt": "max"},
}

"""
 'duration_vm_pu_90_per_line', 'duration_vm_pu_95_per_line', 'duration_vm_pu_97_per_line', 'duration_vm_pu_103_per_line', 'duration_vm_pu_105_per_line', 'duration_vm_pu_110_per_line',
"""

calculated_metrics = {
    "vm_pu_min smaller 0.97": {"label": "$V_\mathrm{min}$ < 0.97 p.u. in %/a"},
    "vm_pu_min smaller 0.9": {"label": "$V_\mathrm{min}$ < 0.9 p.u. in %/a"},
    "vm_pu_min smaller 0.95": {"label": "$V_\mathrm{min}$ < 0.95 p.u. in %/a"},
    "percent_max_trafo_load in %": {"label": "Maximal Transformer Load in %"}
}


def get_percent_smaller_than(np_array, threshold):
    return np.count_nonzero(np_array < threshold) / len(np_array) * 100


def load_case_and_trafo_data(path: Path, quota_variation: QuotaVariation, grid: str):
    with open(path.joinpath("results_to_plot.json"), "r") as file:
        results = json.load(file)
    case_and_tafo_data = {}
    name_value_dict = quota_variation.get_quota_case_name_and_value_dict()
    for case, case_data in results["grid"].items():
        trafo_data = {}
        startswith = f"results_{case}_{grid}_3-ph_"
        for file_name in os.listdir(path):
            if file_name.startswith(startswith) and file_name.endswith(".json"):
                trafo_size = int(file_name.replace(startswith, "").replace(".json", ""))
                with open(path.joinpath(file_name), "r") as file:
                    trafo_results = json.load(file)
                trafo_results_converted = {}
                for metric, values in trafo_results.items():
                    if isinstance(values, list):
                        trafo_results_converted[metric] = np.array(values)
                    else:
                        trafo_results_converted[metric] = values
                metrics = extract_detailed_grid_info(trafo_results_converted)
                trafo_results_converted.update(metrics)
                trafo_data[trafo_size] = trafo_results_converted
        case_and_tafo_data[name_value_dict[case]] = trafo_data
    return case_and_tafo_data


def extract_detailed_grid_info(trafo_results: dict):
    grid_metrics = {}
    for metric, metric_values in trafo_results.items():
        if not isinstance(metric_values, dict):
            continue
        if "bus" in next(iter(metric_values)):
            id_street = "bus"
            id_house = "loadbus"
        else:
            id_street = "line"
            id_house = "branchout_line"
        n_branches = 10
        branch_order = [9, 4, 5, 1, 2, 3, 6, 7, 8, 10]
        n_houses_max = 32
        df = pd.DataFrame(columns=range(1, n_houses_max + 1))
        for branch_number in branch_order:
            for house_number in range(1, n_houses_max + 1):
                try:
                    value_house = metric_values[f"{id_house}_{branch_number}_{house_number}"]
                    value_street = metric_values[f"{id_street}_{branch_number}_{house_number}"]
                except KeyError:
                    value_house = np.NAN
                    value_street = np.NAN
                df.loc[f"{branch_number}_house", house_number] = value_house
                df.loc[f"{branch_number}_street", house_number] = value_street
        grid_metrics[metric] = df
    return grid_metrics


def plot_time_series(
        case_trafo_data: dict,
        metric: str, quota_variation: QuotaVariation,
        save_path, fixed_case: str = None, fixed_trafo_size: int = None):
    if fixed_case is None and fixed_trafo_size is None:
        raise ValueError("One of both case or trafo_size must be fixed")
    if fixed_case is not None and fixed_trafo_size is not None:
        raise ValueError("Not both case or trafo_size can be fixed")
    if isinstance(quota_variation.varying_technologies, dict):
        varying_tech_name = quota_variation.pretty_name(
            quota_variation.get_single_varying_technology_name_and_quotas()[0]
        )
    else:
        varying_tech_name = ""

    if fixed_trafo_size is None:
        _data = {f"{key} kVA": case_trafo_data[fixed_case][key][metric]
                 for key in sorted(case_trafo_data[fixed_case].keys())}
        fig_title = "Transformer variation"
        if varying_tech_name:
            fig_title += f" | {varying_tech_name}-quota={fixed_case}"
        save_name = f"case={fixed_case}"
    else:
        _data = {case: trafo_data[fixed_trafo_size][metric] for case, trafo_data in case_trafo_data.items()}
        fig_title = f"{fixed_trafo_size} kVA transformer"
        if varying_tech_name:
            fig_title = f"{varying_tech_name}-quota variation | " + fig_title
        save_name = f"trafo={fixed_trafo_size}"

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=get_figure_size(n_columns=2))
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
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[0])
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[1])

    fig.suptitle(fig_title)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path.joinpath(f"{metric}_{save_name}.png"))
    plt.close("all")


def get_statistics(case_and_trafo_data: dict, path: Path, quota_variation: QuotaVariation):
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
    df.loc[:, "percent_max_trafo_load in %"] = df.loc[:, "s_trafo_max"] / df.loc[:, "Trafo-Size"] * 100
    df.to_excel(path.joinpath(f"grid_statistics_{path.name}.xlsx"))

    for second_metric in calculated_metrics.keys():
        plot_required_trafo_size(
            path=path.joinpath("plots"),
            df=df,
            quota_variation=quota_variation,
            design_metric="percent_max_trafo_load in %",
            design_value=100,
            second_metric=second_metric
        )

    return df


def plot_all_metrics_and_trafos(case_and_trafo_data: dict, path: Path, quota_variation: QuotaVariation):
    cases = list(case_and_trafo_data.keys())
    trafo_sizes = list(case_and_trafo_data[cases[0]])
    save_path = path.joinpath("plots")
    os.makedirs(save_path, exist_ok=True)
    for metric in metric_data.keys():
        only_one_trafo_is_enough = metric in ["p_trafo", "s_trafo", "q_trafo"]
        #only_one_trafo_is_enough = False
        for trafo_size in trafo_sizes:
            plot_time_series(case_trafo_data=case_and_trafo_data,
                             save_path=save_path, quota_variation=quota_variation,
                             metric=metric, fixed_trafo_size=trafo_size)
            if only_one_trafo_is_enough:
                break
        for case in cases:
            plot_time_series(case_trafo_data=case_and_trafo_data,
                             save_path=save_path, quota_variation=quota_variation,
                             metric=metric, fixed_case=case)


def generate_all_cases(path: Path, with_plot: bool, altbau=True):
    if altbau:
        grid_case = "Altbau_"
        grid_str = "altbau"
    else:
        grid_case = "Neubau_"
        grid_str = "neubau"

    from hps_grid_interaction.monte_carlo.monte_carlo import get_all_quota_studies
    all_quota_cases = get_all_quota_studies()
    dfs = []
    folders = [
        folder
        for folder in os.listdir(path)
        if folder.startswith(grid_case) and os.path.isdir(path.joinpath(folder))
    ]
    for idx, folder in enumerate(folders):
        case = folder.replace(grid_case, "")
        case_path = path.joinpath(folder)
        quota_variation = all_quota_cases[case]
        case_and_trafo_data = load_case_and_trafo_data(case_path, quota_variation=quota_variation, grid=grid_str)
        if with_plot:
            plot_all_metrics_and_trafos(case_and_trafo_data, case_path, quota_variation)
            plot_grid_as_heatmap(case_and_trafo_data, case_path)
        df = get_statistics(case_and_trafo_data, case_path, quota_variation)
        df.loc[:, "quota_cases"] = case
        dfs.append(df)
        print(f"Ran {folder} ({idx + 1}/{len(folders)})")
    dfs = pd.concat(dfs)
    dfs.index = range(len(dfs))
    dfs.to_excel(path.joinpath(f"{grid_case}all_grid_results_ex.xlsx"))


def set_color_of_axis(axis, color: str):
    axis.label.set_color(color)
    [t.set_color(color) for t in axis.get_ticklines()]
    [t.set_color(color) for t in axis.get_ticklabels()]


def plot_required_trafo_size(
        path: Path,
        df: pd.DataFrame(),
        quota_variation: QuotaVariation,
        design_metric: str = "percent_max_trafo_load in %",
        design_value: float = 100,
        second_metric: str = "vm_pu_min smaller 0.97"
):
    df_plot = pd.DataFrame(columns=df.columns)
    for case in quota_variation.get_varying_technology_ids():
        df_case = df.loc[(df.loc[:, "case"] == case) & (df.loc[:, design_metric] < design_value)]
        min_idx = df_case.loc[:, "Trafo-Size"].argmin()
        df_plot.loc[case] = df_case.iloc[min_idx]
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1, height_factor=1.3))
    ax_twin = ax.twiny()
    x_pos = np.arange(len(df_plot.index))
    bar_width = 0.8 / 1
    bar_args = dict(align='center', ecolor='black', height=bar_width)
    ax.barh(
        x_pos, df_plot.loc[:, "Trafo-Size"],
        color=EBCColors.ebc_palette_sort_2[0],
        **bar_args,
    )
    ax.set_xlabel("Minimal Transformer Size in kVA")
    ax.set_yticks(x_pos)
    ax.xaxis.grid(True)
    ax.set_xlim([df_plot.loc[:, "Trafo-Size"].min()-100, df_plot.loc[:, "Trafo-Size"].max()+100])
    ax_twin.plot(df_plot.loc[:, second_metric], x_pos, linewidth=5, color=EBCColors.ebc_palette_sort_2[1])
    ax_twin.set_xlabel(calculated_metrics[second_metric]["label"])
    set_color_of_axis(axis=ax_twin.xaxis, color=EBCColors.ebc_palette_sort_2[1])

    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax, which_axis="y", title_offset=0.2)
    ax.set_yticklabels(df_plot.index)
    set_color_of_axis(axis=ax.xaxis, color=EBCColors.ebc_palette_sort_2[0])

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig(path.joinpath(f"MinTrafoDesign_{second_metric}.png"))


def plot_grid_as_heatmap(case_and_trafo_data: dict, save_path: Path):
    import seaborn as sns
    save_path = save_path.joinpath("plots_detailed_grid")
    os.makedirs(save_path, exist_ok=True)
    for case, case_data in case_and_trafo_data.items():
        for trafo_size, trafo_data in case_data.items():
            for metric, df in trafo_data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                metric_name = metric.replace("_per_line", "")
                if metric_name not in metric_data:
                    continue
                metric_kwargs = metric_data.get(metric_name, {})
                fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1, height_factor=0.7))
                sns.heatmap(df.astype(float), ax=ax, cmap='rocket', linewidths=0,
                            zorder=1, linecolor='black', **metric_kwargs.get("min_max", {}))
                ax.set_title(metric_kwargs.get("label", metric_name))
                ax.imshow(plt.imread(DATA_PATH.joinpath("altbau_grid.png"), format="png"),
                          aspect=ax.get_aspect(),
                          extent=ax.get_xlim() + ax.get_ylim(),
                          zorder=2
                          )
                plt.axis('off')
                fig.tight_layout()
                fig.savefig(save_path.joinpath(f"{case}_{metric}.png"))
                plt.close("all")


if __name__ == '__main__':
    PlotConfig.load_default()
    from hps_grid_interaction import RESULTS_MONTE_CARLO_FOLDER
    #PATH = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\03_monte_carlo")
    generate_all_cases(RESULTS_MONTE_CARLO_FOLDER, with_plot=True)
