import itertools
import os
import json
import pickle
from pathlib import Path
import ast

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.monte_carlo.monte_carlo import QuotaVariation
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size, icon_plotting
from hps_grid_interaction.monte_carlo.plots import plot_quota_case_with_images
from hps_grid_interaction import DATA_PATH

METRIC_DATA = {
    "p_trafo": {"label": "$P$ in kW", "opt": "max", "label_abs": "$|P|$ in kW"},
    "q_trafo": {"label": "$Q$ in kW", "opt": "max"},
    "s_trafo": {"label": "$S_\mathrm{tf}$ in kVA", "opt": "max", "label_abs": "$|S_\mathrm{tf}|$ in kVA"},
    "vm_pu_min": {"label": "$V_\mathrm{min}$ in p.u.", "opt": "min",
                  "axhlines": [0.9, 0.95, 0.97], "min_max": {"vmin": 0.9, "vmax": 1}},
    #    "vm_pu_max": {"label": "$V_\mathrm{max}$ in p.u.", "opt": "max", "min_max": {"vmin": 0.85, "vmax": 1}},
    "max_line_loading": {"label": "$L_\mathrm{max}$ in %", "opt": "max", "min_max": {"vmin": 0, "vmax": 100}},
}

"""
 'duration_vm_pu_90_per_line', 'duration_vm_pu_95_per_line', 'duration_vm_pu_97_per_line', 'duration_vm_pu_103_per_line', 'duration_vm_pu_105_per_line', 'duration_vm_pu_110_per_line',
"""

CALCULATED_METRICS = {
    "vm_pu_min smaller 0.97": {"label": "$V_\mathrm{min}$ < 0.97 p.u. in %/a"},
    "vm_pu_min smaller 0.9": {"label": "$V_\mathrm{min}$ < 0.9 p.u. in %/a"},
    "vm_pu_min smaller 0.95": {"label": "$V_\mathrm{min}$ < 0.95 p.u. in %/a"},
    "percent_max_trafo_load in %": {"label": "Maximal Transformer Load in %"},
    "max_line_loading_max": {"label": "$L_\mathrm{max}$ in %"}
}

MAIN_MC_METRIC = "argmean"
MAX_MC_METRIC = "percentile_997"
MIN_MC_METRIC = "percentile_03"

MONTE_CARLO_METRICS = {
    "main": MAIN_MC_METRIC,
    "max": MAX_MC_METRIC,
    "min": MIN_MC_METRIC
}


def get_percent_smaller_than(np_array, threshold):
    return np.count_nonzero(np_array < threshold) / len(np_array) * 100


def load_case_and_trafo_data(path: Path, quota_variation: QuotaVariation, grid: str):
    with open(path.joinpath("results_to_plot.json"), "r") as file:
        results = json.load(file)
    case_and_transformer_data = {}
    name_value_dict = quota_variation.get_quota_case_name_and_value_dict()
    for monte_carlo_metric in MONTE_CARLO_METRICS.values():
        for case, case_data in results[monte_carlo_metric]["grid"].items():
            case_name = name_value_dict[case]
            startswith = f"results_{case}_{grid}_3-ph_"
            json_result_files = [path.joinpath(monte_carlo_metric, file_name)
                                 for file_name in os.listdir(path.joinpath(monte_carlo_metric))
                                 if file_name.startswith(startswith) and file_name.endswith(".json")]
            trafo_data = case_and_transformer_data.get(case_name, {})
            for json_result_file in json_result_files:
                trafo_size = int(json_result_file.stem.replace(startswith, ""))
                with open(json_result_file, "r") as file:
                    trafo_results = json.load(file)
                trafo_results_converted = {}
                for metric, values in trafo_results.items():
                    if isinstance(values, list):
                        trafo_results_converted[metric] = np.array(values)
                    else:
                        trafo_results_converted[metric] = values
                metrics = extract_detailed_grid_info(trafo_results_converted)
                trafo_results_converted.update(metrics)
                if trafo_size in trafo_data:
                    trafo_data[trafo_size][monte_carlo_metric] = trafo_results_converted
                else:
                    trafo_data[trafo_size] = {monte_carlo_metric: trafo_results_converted}
            case_and_transformer_data[case_name] = trafo_data
    return case_and_transformer_data


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


def convert_grid_df_to_heatmap_df(df_grid: pd.DataFrame, column):
    df_grid = df_grid.copy()
    df_grid = df_grid.set_index("Anschlusspunkt")
    branch_order = [9, 4, 5, 1, 2, 3, 6, 7, 8, 10]
    n_houses_max = 32
    df = pd.DataFrame(columns=range(1, n_houses_max + 1))
    for branch_number in branch_order:
        for house_number in range(1, n_houses_max + 1):
            try:
                value_house = df_grid.loc[f"{branch_number}-{house_number}", column]
            except KeyError:
                value_house = np.NAN
            value_street = np.NAN
            df.loc[f"{branch_number}_house", house_number] = value_house
            df.loc[f"{branch_number}_street", house_number] = value_street
    return df


def plot_time_series(
        case_and_trafo_data: dict,
        metric: str,
        quota_variation: QuotaVariation,
        save_path,
        fixed_case: str = None,
        fixed_trafo_size: int = None,
):
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
        fixed_case_data = case_and_trafo_data[fixed_case]
        _data = {
            f"{key} kVA": {
                monte_carlo_metric: fixed_case_data[key][monte_carlo_metric][metric]
                for monte_carlo_metric in fixed_case_data[key].keys()
            } for key in sorted(fixed_case_data.keys())
        }
        fig_title = "Transformer variation"
        if varying_tech_name:
            fig_title += f" | {varying_tech_name}-quota={fixed_case}"
        save_name = f"case={fixed_case}"
    else:
        _data = {
            case: {
                monte_carlo_metric: trafo_results[metric]
                for monte_carlo_metric, trafo_results in trafo_data[fixed_trafo_size].items()
            } for case, trafo_data in case_and_trafo_data.items()
        }
        fig_title = f"{fixed_trafo_size} kVA transformer"
        if varying_tech_name:
            if varying_tech_name == "Hybrid":
                fig_title = f"Hybrid HP share variation | " + fig_title
            elif varying_tech_name == "Retrofit":
                fig_title = "$\it{retrofit}$ rate variation | " + fig_title
            elif varying_tech_name == "Advanced-retrofit":
                fig_title = "$\it{advanced-retrofit}$ rate variation | " + fig_title
            else:
                raise ValueError(f"Given varying tech name not supported: {varying_tech_name}")
        save_name = f"trafo={fixed_trafo_size}"
    different_y_labels = metric in ["s_trafo", "p_trafo"]
    fig, ax = plt.subplots(
        1, 2,
        sharey=not different_y_labels,
        figsize=get_figure_size(n_columns=2, height_factor=1.2)
    )
    t_oda = load_outdoor_air_temperature()
    bins = np.linspace(t_oda.values[1:-1, 0].min(), t_oda.values[1:-1, 0].max(), num=30)
    categories = pd.cut(t_oda.values[1:-1, 0], bins, labels=False)
    idx_case = 0
    max_oda_data = {}
    overall_max = 0
    for _label, _monte_carlo_tsd in _data.items():
        curves = {}
        for monte_carlo_metric, time_series_data in _monte_carlo_tsd.items():
            curve = []
            for bin_idx in range(len(bins)):
                bin_mask = (categories == bin_idx)
                if np.any(bin_mask):
                    if METRIC_DATA[metric]["opt"] == "min":
                        curve.append(np.abs(time_series_data[bin_mask]).min())
                    else:
                        curve.append(np.abs(time_series_data[bin_mask]).max())
                else:
                    curve.append(np.NAN)
            curves[monte_carlo_metric] = curve
        main_tsd = _monte_carlo_tsd[MAIN_MC_METRIC]
        max_tsd = _monte_carlo_tsd[MAX_MC_METRIC]
        min_tsd = _monte_carlo_tsd[MIN_MC_METRIC]
        all_curves = np.array([curves[MIN_MC_METRIC], curves[MAX_MC_METRIC], curves[MAIN_MC_METRIC]])
        main_cluster = curves[MAIN_MC_METRIC]
        max_cluster = np.max(all_curves, axis=0)
        min_cluster = np.min(all_curves, axis=0)
        color = EBCColors.ebc_palette_sort_2[idx_case]
        x_year = np.arange(len(main_tsd)) / 4
        uncertainty_kwargs = dict(
            edgecolor=None, alpha=0.5, facecolor=color
        )
        ax[0].plot(x_year, np.sort(main_tsd)[::-1], label=_label, color=color)
        ax[0].fill_between(x_year, np.sort(min_tsd)[::-1], np.sort(max_tsd)[::-1], **uncertainty_kwargs)
        ax[1].plot(bins, main_cluster, label=_label, linestyle="-")
        ax[1].fill_between(bins, min_cluster, max_cluster, **uncertainty_kwargs)

        idx_case += 1
        argmax_cluster = np.nanargmax(np.array(main_cluster))
        max_oda_data[_label] = {
            "t_oda": bins[argmax_cluster],
            MAIN_MC_METRIC: main_cluster[argmax_cluster],
            MAX_MC_METRIC: max_cluster[argmax_cluster],
            MIN_MC_METRIC: min_cluster[argmax_cluster]
        }
        overall_max = max(overall_max, np.max(np.abs(max_tsd)))

    axhlines = METRIC_DATA[metric].get("axhlines", None)
    if axhlines is not None:
        for hline in axhlines:
            for _ax in ax:
                _ax.axhline(hline, color="black")
    fig.suptitle(fig_title)
    y_label_non_abs = METRIC_DATA[metric]["label"]
    ax[0].set_ylabel(y_label_non_abs)
    if different_y_labels:
        ax[1].set_ylabel(METRIC_DATA[metric].get("label_abs", y_label_non_abs))
    ax[1].set_xlabel("$T_\mathrm{oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1, borderaxespad=0.02)
    if metric in ["s_trafo", "p_trafo"]:
        curr_ytick_max = overall_max * 1.1
        # ax[1].set_ylim([0, curr_ytick_max])
        # ax[1].set_yticks(np.linspace(0, curr_ytick_max, 7))
        if main_tsd.min() < 0:
            ax[0].set_ylim([-curr_ytick_max, curr_ytick_max])
            # ax[0].set_yticks(np.linspace(-curr_ytick_max, curr_ytick_max, 7))
        else:
            ax[0].set_ylim([0, curr_ytick_max])
            # ax[0].set_yticks(np.linspace(0, curr_ytick_max, 7))
    elif metric in ["vm_pu_min"]:
        for _ax in ax:
            _ax.set_yticks([0.9, 0.92, 0.94, 0.96, 0.98, 1])
    kwargs = dict(width=0.18, title_offset=0.02, distance_to_others=0.02)

    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[0], **kwargs)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[1], **kwargs)

    fig.suptitle(fig_title)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path.joinpath(f"{metric}_{save_name}.png"))
    plt.close("all")
    return max_oda_data


def get_statistics(case_and_trafo_data: dict, path: Path):
    df = pd.DataFrame()
    idx = 0
    for case, trafo_data in case_and_trafo_data.items():
        for trafo_size, trafo_results in trafo_data.items():
            for monte_carlo_metric, trafo_results_monte_carlo in trafo_results.items():
                df.loc[idx, "Trafo-Size"] = trafo_size
                df.loc[idx, "case"] = case
                df.loc[idx, "Monte-Carlo Metric"] = monte_carlo_metric
                for metric, settings in METRIC_DATA.items():
                    opt = settings["opt"]
                    if opt == "min":
                        df.loc[idx, f"{metric}_{opt}"] = np.abs(trafo_results_monte_carlo[metric]).min()
                    if opt == "max":
                        df.loc[idx, f"{metric}_{opt}"] = np.abs(trafo_results_monte_carlo[metric]).max()
                    if "axhlines" in settings:
                        for axhline_value in settings["axhlines"]:
                            df.loc[idx, f"{metric} smaller {axhline_value}"] = get_percent_smaller_than(
                                trafo_results_monte_carlo[metric], axhline_value
                            )
                idx += 1
    df.loc[:, "percent_max_trafo_load in %"] = df.loc[:, "s_trafo_max"] / df.loc[:, "Trafo-Size"] * 100
    df.to_excel(path.joinpath(f"grid_statistics_{path.name}.xlsx"))
    return df


def plot_all_metrics_and_trafos(
        case_and_trafo_data: dict,
        path: Path,
        quota_variation: QuotaVariation
):
    cases = list(case_and_trafo_data.keys())
    trafo_sizes = list(case_and_trafo_data[cases[0]])
    save_path = path.joinpath("plots")
    os.makedirs(save_path, exist_ok=True)
    s_max_cluster = {}
    for metric in METRIC_DATA.keys():
        kwargs = dict(
            case_and_trafo_data=case_and_trafo_data,
            metric=metric,
            save_path=save_path,
            quota_variation=quota_variation
        )
        for trafo_size in trafo_sizes:
            plot_time_series(fixed_trafo_size=trafo_size, **kwargs)
        for case in cases:
            max_oda_data = plot_time_series(fixed_case=case, **kwargs)
            if metric == "s_trafo":
                s_max_cluster[case] = max_oda_data
    return s_max_cluster


def generate_all_cases(
        path: Path, with_plot: bool, oldbuildings=True, use_mp: bool = True
):
    if oldbuildings:
        grid_case = "oldbuildings_"
        grid_str = "oldbuildings"
    else:
        grid_case = "newbuildings_"
        grid_str = "newbuildings"

    from hps_grid_interaction.monte_carlo.monte_carlo import get_all_quota_studies
    all_quota_cases = get_all_quota_studies()
    dfs = []
    folders = [
        folder
        for folder in os.listdir(path)
        if (
                folder.startswith(grid_case)
                and os.path.isdir(path.joinpath(folder))
                and "Analyse" not in folder
                and (
                        folder.endswith("hybrid_PVBat_EMob_HP") or
                        folder.endswith("hybrid_EMob_HP")
                )
        )
    ]
    kwargs_mp = []
    for folder in folders:
        case = folder.replace(grid_case, "")
        kwargs_mp.append(dict(
            case_path=path.joinpath(folder),
            case=case,
            quota_variation=all_quota_cases[case],
            grid_str=grid_str, with_plot=with_plot
        ))

    idx = 0
    dfs_min_trafo_size = []
    s_max_cluster_all_cases = {}
    if use_mp:
        import multiprocessing as mp
        pool = mp.Pool(processes=20)

        for df, df_min_trafo_size, _s_max_cluster_all_cases in pool.imap_unordered(create_plots_and_get_df, kwargs_mp):
            dfs.append(df)
            dfs_min_trafo_size.append(df_min_trafo_size)
            s_max_cluster_all_cases.update(_s_max_cluster_all_cases)
            print(f"Ran {idx + 1}/{len(folders)} folders")
            idx += 1
    else:
        for kwargs in kwargs_mp:
            df, df_min_trafo_size, _s_max_cluster_all_cases = create_plots_and_get_df(kwargs)
            dfs.append(df)
            dfs_min_trafo_size.append(df_min_trafo_size)
            s_max_cluster_all_cases.update(_s_max_cluster_all_cases)
            print(f"Ran {idx + 1}/{len(folders)} folders:", kwargs)
            idx += 1
    dfs = pd.concat(dfs)
    for metric in MONTE_CARLO_METRICS.values():
        df_min_trafo_size = pd.concat([dfs[metric] for dfs in dfs_min_trafo_size])
        df_min_trafo_size.to_excel(path.joinpath(f"{grid_case}minimal_trafo_sizes_{metric}.xlsx"))
    dfs.index = range(len(dfs))
    dfs.to_excel(path.joinpath(f"{grid_case}all_grid_results_ex.xlsx"))
    with open(path.joinpath(f"{grid_case}max_data.json"), "w+") as file:
        json.dump(s_max_cluster_all_cases, file)


def create_plots_and_get_df(kwargs):
    # Load RC Params
    PlotConfig.load_default()

    quota_variation = kwargs["quota_variation"]
    grid_str = kwargs["grid_str"]
    case_path = kwargs["case_path"]
    case = kwargs["case"]
    with_plot = kwargs.get("with_plot", True)

    print(f"Extracting and plotting {case_path}")
    s_max_cluster_all_cases = {}
    try:
        case_and_trafo_data = load_case_and_trafo_data(case_path, quota_variation=quota_variation, grid=grid_str)
        df = get_statistics(case_and_trafo_data=case_and_trafo_data, path=case_path)
        df.loc[:, "quota_cases"] = case
        if with_plot:
            s_max_cluster = plot_all_metrics_and_trafos(
                case_and_trafo_data, case_path, quota_variation
            )
            s_max_cluster_all_cases[case] = s_max_cluster
            for monte_carlo_metric in MONTE_CARLO_METRICS.values():
                pass
                # plot_grid_as_heatmap_one_big_image(case_and_trafo_data, case_path, monte_carlo_metric=monte_carlo_metric)
                # plot_grid_as_heatmap_single_images(case_and_trafo_data, case_path, monte_carlo_metric=monte_carlo_metric)
        trafo_sizes_metrics = {}
        for monte_carlo_metric in MONTE_CARLO_METRICS.values():
            trafo_sizes_metrics[monte_carlo_metric] = plot_required_trafo_size(
                path=case_path.joinpath("plots"),
                df=df,
                quota_variation=quota_variation,
                design_metric="percent_max_trafo_load in %",
                design_value=100,
                monte_carlo_metric=monte_carlo_metric,
                without_plot=True
            )

        print(f"Extracted and plotted {case_path}")
        return df, trafo_sizes_metrics, s_max_cluster_all_cases
    except Exception as err:
        raise err
        print(f"Error during extraction of {case_path}: {err}")
        return pd.DataFrame()


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
        monte_carlo_metric: str = "argmax",
        without_plot: bool = False
):
    os.makedirs(path, exist_ok=True)
    df_plot = pd.DataFrame(columns=df.columns)
    for case in quota_variation.get_varying_technology_ids():
        df_case = df.loc[
            (df.loc[:, "case"] == case) &
            (df.loc[:, "Monte-Carlo Metric"] == monte_carlo_metric) &
            (df.loc[:, design_metric] < design_value)
            ]
        try:
            min_idx = df_case.loc[:, "Trafo-Size"].argmin()
            df_plot.loc[case] = df_case.iloc[min_idx]
        except ValueError:
            df_plot.loc[case] = np.NAN
    df_plot = df_plot.iloc[::-1]
    df_plot.loc[:, "fixed_technologies"] = df_plot.loc[:, "Trafo-Size"].apply(
        lambda x: quota_variation.fixed_technologies)

    if without_plot:
        return df_plot
    from copy import deepcopy
    quota_variation_copy = deepcopy(quota_variation)
    if isinstance(quota_variation.varying_technologies, dict):
        name, values = quota_variation.get_single_varying_technology_name_and_quotas()
        quota_variation_copy.varying_technologies = {name: values[::-1]}
    else:
        quota_variation_copy.varying_technologies = quota_variation_copy.varying_technologies[::-1]

    for second_metric, plot_settings in CALCULATED_METRICS.items():
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
        ax.set_xlabel("Minimum Transformer Size in kVA")
        ax.set_yticks(x_pos)
        ax.xaxis.grid(True)
        ax.set_xlim([df_plot.loc[:, "Trafo-Size"].min() - 100, df_plot.loc[:, "Trafo-Size"].max() + 100])
        ax_twin.plot(df_plot.loc[:, second_metric], x_pos, linewidth=5, color=EBCColors.ebc_palette_sort_2[1])
        ax_twin.set_xlabel(plot_settings["label"])
        set_color_of_axis(axis=ax_twin.xaxis, color=EBCColors.ebc_palette_sort_2[1])
        plot_quota_case_with_images(quota_variation=quota_variation_copy, ax=ax, which_axis="y", title_offset=0.2)
        # ax.set_yticklabels(df_plot.index)
        set_color_of_axis(axis=ax.xaxis, color=EBCColors.ebc_palette_sort_2[0])

        fig.tight_layout(pad=0.15, w_pad=0.1, h_pad=0.1)
        fig.savefig(path.joinpath(f"MinTrafoDesign_{second_metric}_{design_metric}.png"))
    return df_plot


def plot_grid_as_heatmap_one_big_image(case_and_trafo_data: dict, save_path: Path, monte_carlo_metric: str = "argmean"):
    save_path = save_path.joinpath("plots_detailed_grid")
    os.makedirs(save_path, exist_ok=True)
    n_trafo_sizes = len(next(iter(case_and_trafo_data.values())))
    n_cases = len(case_and_trafo_data)
    figs, axes = [], []

    metrics = [
        "max_line_loading_per_line",
        "vm_pu_min_per_line"
    ]
    for _ in metrics:
        fig, ax = plt.subplots(n_cases, n_trafo_sizes,
                               figsize=get_figure_size(n_columns=1 * n_trafo_sizes, height_factor=0.7 * n_cases),
                               sharex=True, sharey=True)
        figs.append(fig)
        axes.append(ax)

    for idx_case, case in enumerate(case_and_trafo_data.keys()):
        case_data = case_and_trafo_data[case]
        for idx_trafo, tafo_size in enumerate(sorted(case_data.keys())):
            trafo_data = case_data[tafo_size][monte_carlo_metric]
            for metric, ax in zip(metrics, axes):
                df = trafo_data[metric]
                if not isinstance(df, pd.DataFrame):
                    continue
                metric_name = metric.replace("_per_line", "")
                if metric_name not in METRIC_DATA:
                    continue
                metric_kwargs = METRIC_DATA.get(metric_name, {})
                if metric_kwargs["opt"] == "min":
                    cmap = get_colormap(r=True)
                else:
                    cmap = get_colormap()
                if metric == "vm_pu_min_per_line":
                    cbar_kws = {"ticks": [0.9, 0.92, 0.94, 0.96, 0.98, 1]}
                else:
                    cbar_kws = {}
                ax[idx_case, idx_trafo] = plot_heat_map_on_grid_image(
                    ax=ax[idx_case, idx_trafo], df=df, cbar_kws=cbar_kws,
                    cmap=cmap, heatmap_kwargs=metric_kwargs.get("min_max", {})
                )

    for idx_case, case in enumerate(case_and_trafo_data.keys()):
        for metric, ax in zip(metrics, axes):
            ax[idx_case, 0].set_ylabel(case)

    for idx_trafo, tafo_size in enumerate(sorted(case_and_trafo_data[case].keys())):
        for metric, ax in zip(metrics, axes):
            ax[-1, idx_trafo].set_xlabel(f"{tafo_size} kVA")

    for metric, fig in zip(metrics, figs):
        metric_name = metric.replace("_per_line", "")
        fig.suptitle(METRIC_DATA[metric_name].get("label", metric_name))
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"{metric}_{monte_carlo_metric}.png"))
        plt.close("all")


def plot_grid_as_heatmap_single_images(case_and_trafo_data: dict, save_path: Path, monte_carlo_metric: str = "argmean"):
    save_path = save_path.joinpath("plots_detailed_grid_single")
    os.makedirs(save_path, exist_ok=True)
    metrics = [
        "max_line_loading_per_line",
        "vm_pu_min_per_line"
    ]
    for idx_case, case in enumerate(case_and_trafo_data.keys()):
        case_data = case_and_trafo_data[case]
        for idx_trafo, trafo_size in enumerate(sorted(case_data.keys())):
            trafo_data = case_data[trafo_size][monte_carlo_metric]
            for metric in metrics:
                fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1, height_factor=0.7))
                df = trafo_data[metric]
                if not isinstance(df, pd.DataFrame):
                    continue
                metric_name = metric.replace("_per_line", "")
                if metric_name not in METRIC_DATA:
                    continue
                metric_kwargs = METRIC_DATA.get(metric_name, {})
                if metric_kwargs["opt"] == "min":
                    cmap = get_colormap(r=True)
                else:
                    cmap = get_colormap()
                if metric == "vm_pu_min_per_line":
                    cbar_kws = {"ticks": [0.9, 0.92, 0.94, 0.96, 0.98, 1]}
                else:
                    cbar_kws = {}
                ax = plot_heat_map_on_grid_image(
                    ax=ax, df=df, cbar_kws=cbar_kws,
                    cmap=cmap, heatmap_kwargs=metric_kwargs.get("min_max", {})
                )
                ax.set_ylabel(case)
                ax.set_xlabel(f"{trafo_size} kVA")

                metric_name = metric.replace("_per_line", "")
                fig.suptitle(METRIC_DATA[metric_name].get("label", metric_name))
                fig.tight_layout()
                fig.savefig(save_path.joinpath(f"{metric}_{monte_carlo_metric}_{case}_{trafo_size}.png"))
                plt.close("all")


def plot_heat_map_on_grid_image(ax: plt.axes, cmap, df: pd.DataFrame, cbar_kws: dict, heatmap_kwargs: dict = None):
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    sns.heatmap(df.astype(float), ax=ax, cmap=cmap, linewidths=0,
                zorder=1, linecolor='black', **heatmap_kwargs, cbar_kws=cbar_kws)
    ax.imshow(
        plt.imread(DATA_PATH.joinpath("grid_image.png"), format="png"),
        aspect=ax.get_aspect(),
        extent=ax.get_xlim() + ax.get_ylim(),
        zorder=2
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def aggregate_simultaneity_factors(path: Path):
    folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(path.joinpath(folder)) and not folder.startswith("_")
    ]
    all_factors = []
    for folder in folders:
        json_path = path.joinpath(folder, "simultaneity_factors.json")
        with open(json_path, "r") as file:
            factors = json.load(file)
            for key, value in factors.items():
                all_factors.append({
                    "case": folder, "quota": key.split("_")[-1], "factor": value["factor"]
                })
    df = pd.DataFrame(all_factors)
    df.to_excel(path.joinpath("simultaneity_factors.xlsx"))


def _get_column_name_with_mc(name, mc):
    _d = {"min": 0, "main": 1, "max": 2}
    return f"{name}_{_d[mc]}_{mc}"


def _reindex_with_uncertainty(order):
    new_order = []
    for name in order:
        for mc in ["min", "main", "max"]:
            new_order.append(_get_column_name_with_mc(name, mc))
    return new_order


def plot_heat_map_trafo_size_with_uncertainty(
        path: Path,
        save_path: Path,
        use_case: str = "hybrid",
        oldbuildings: bool = False,
):
    if oldbuildings:
        grid_name = "oldbuildings"
    else:
        grid_name = "newbuildings"

    mc_dfs = {}
    for mc_type, mc_metric in MONTE_CARLO_METRICS.items():
        df = pd.read_excel(path.joinpath(f"{grid_name}_minimal_trafo_sizes_{mc_metric}.xlsx"), index_col=0)
        df.loc[:, "quota"] = df.index
        mask = df.loc[:, "quota_cases"].apply(lambda x: x.startswith(f"{use_case}_"))
        mc_dfs[mc_type] = df.loc[mask]
    retro_order = [
        "['heat_pump']",
        "['pv', 'battery', 'heat_pump']",
        "['e_mobility', 'heat_pump']",
        "['pv', 'battery', 'e_mobility', 'heat_pump']",
        "['heat_pump', 'heating_rod']",
        "['pv', 'battery', 'heat_pump', 'heating_rod']",
        "['e_mobility', 'heat_pump', 'heating_rod']",
        "['pv', 'battery', 'e_mobility', 'heat_pump', 'heating_rod']",
        "['heat_pump', 'hybrid']",
        "['pv', 'battery', 'heat_pump', 'hybrid']",
        "['e_mobility', 'heat_pump', 'hybrid']",
        "['pv', 'battery', 'e_mobility', 'heat_pump', 'hybrid']",
    ]
    special_orders = {
        "hybrid": [
            "['average', 'heat_pump']",
            "['average', 'pv', 'heat_pump']",
            "['average', 'e_mobility', 'heat_pump']",
            "['average', 'pv', 'battery', 'heat_pump']",
            "['average', 'pv', 'e_mobility', 'heat_pump']",
            "['average', 'pv', 'battery', 'e_mobility', 'heat_pump']",
            "['average', 'heat_pump', 'heating_rod']",
            "['average', 'pv', 'heat_pump', 'heating_rod']",
            "['average', 'e_mobility', 'heat_pump', 'heating_rod']",
            "['average', 'pv', 'battery', 'heat_pump', 'heating_rod']",
            "['average', 'pv', 'e_mobility', 'heat_pump', 'heating_rod']",
            "['average', 'pv', 'battery', 'e_mobility', 'heat_pump', 'heating_rod']",
        ], "retrofit": retro_order, "adv_retrofit": retro_order
    }

    # df.index = range(len(df))
    metrics_to_plot = {
        "Trafo-Size": "Minimum Transformer Size in kVA",
        **{metric: kwargs["label"] for metric, kwargs in CALCULATED_METRICS.items()}
    }
    os.makedirs(save_path.joinpath("heatmap_tables"), exist_ok=True)
    for metric, title in metrics_to_plot.items():
        kwargs = dict(
            metric=metric, use_case=use_case, orders=special_orders[use_case],
            title=title, save_path=save_path
        )
        joined_heatmap = pd.DataFrame()
        total_min = min(df.loc[:, metric].min() for df in mc_dfs.values())
        total_max = max(df.loc[:, metric].max() for df in mc_dfs.values())
        for mc, df in mc_dfs.items():
            heatmap = df.pivot(index='quota', columns='fixed_technologies', values=metric)
            joined_heatmap.index = heatmap.index
            for column in heatmap.columns:
                joined_heatmap.loc[:, _get_column_name_with_mc(column, mc)] = heatmap.loc[:, column]
            _plot_single_heat_map_trafo_size(
                **kwargs,
                heatmap=heatmap,
                save_name=f"{grid_name}_{use_case}_{mc}_{metric}",
                with_uncertainty=False,
                vmin=total_min, vmax=total_max
            )
        _plot_single_heat_map_trafo_size(
            **kwargs,
            heatmap=joined_heatmap,
            save_name=f"{grid_name}_{use_case}_{metric}",
            with_uncertainty=True,
            vmin=joined_heatmap.values.min(), vmax=joined_heatmap.values.max()
        )


def get_colormap(r: bool = False):
    from matplotlib.colors import LinearSegmentedColormap
    if r:
        return LinearSegmentedColormap.from_list("custom_cmap_r", EBCColors.ebc_palette_sort_3[::-1], N=500)
    return LinearSegmentedColormap.from_list("custom_cmap", EBCColors.ebc_palette_sort_3, N=500)


def _plot_single_heat_map_trafo_size(
        heatmap: pd.DataFrame, save_path: Path, metric: str,
        use_case: str, orders: list, title: str,
        with_uncertainty: bool, save_name: str,
        vmin: float = None, vmax: float = None
):
    trafo_sizes = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    cbar_kws = {'pad': 0.02}
    if metric == "Trafo-Size":
        unique_labels = pd.unique(heatmap.values.ravel())
        try:
            unique_labels = np.arange(vmin, vmax + 1, 200)
        except Exception:
            print("Hasdas")
        boundaries = [size - 100 for size in trafo_sizes] + [trafo_sizes[-1] + 100]
        color_palette = EBCColors.ebc_palette_sort_3[:len(trafo_sizes)]
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(color_palette)
        # Define the boundaries for the colormap
        norm = BoundaryNorm(boundaries, cmap.N)
        cbar_kws['ticks'] = trafo_sizes
    else:
        cmap = get_colormap()
        norm = None

    kwargs = dict(vmax=vmax, vmin=vmin)

    heatmap = heatmap.astype(float)
    heatmap.index = heatmap.index.map(lambda x: int(x.replace("%", "")))
    heatmap = heatmap.sort_index()
    if with_uncertainty:
        heatmap = heatmap.reindex(columns=_reindex_with_uncertainty(orders))
    else:
        heatmap = heatmap.reindex(columns=orders)

    heatmap.to_excel(save_path.joinpath("heatmap_tables", f"{save_name}.xlsx"))

    fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=2, height_factor=1.5))

    sns.heatmap(heatmap, ax=ax, linewidths=0.5, cmap=cmap, norm=norm,
                zorder=1, linecolor='black', cbar_kws=cbar_kws, **kwargs)

    if metric == "Trafo-Size":
        add_discrete_colorbar(ax=ax, labels=sorted(unique_labels))

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    varying_technologies = [ast.literal_eval(col.replace("'average', ", "")) for col in orders]
    if with_uncertainty:
        ax.set_xticks(np.arange(1.5, len(heatmap.columns) + 0.5, 3))
        for idx in range(0, len(heatmap.columns), 3):
            ax.axvline(idx, color="black")
        for idx in range(0, len(heatmap.index), 1):
            ax.axhline(idx, color="black")

    icon_plotting.add_images_to_axis(
        technologies=varying_technologies, ax=ax,
        which_axis="x", width=0.08,
        distance_to_others=0.02
    )
    icon_plotting.add_image_and_text_as_label(
        ax=ax, which_axis="y", technology=use_case, width=0.12,
        ticklabels=[f"{i}%" for i in heatmap.index],
        distance_to_others=0.12
    )
    ax.set_title(title)

    if metric == "Trafo-Size":
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(trafo_sizes)
        cbar.set_ticklabels(trafo_sizes)
        cbar.set_ticks([], minor=True)

    fig.tight_layout(pad=0.5, rect=[0, -0.06, 1, 1], w_pad=0, h_pad=0)
    fig.savefig(save_path.joinpath(f"{save_name}.png"))
    plt.close("all")


def add_discrete_colorbar(ax, labels):
    n = len(labels)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + (i + 0.5) * r / n for i in range(n)])
    colorbar.set_ticklabels(labels)


def plot_all_heat_map_trafo_size(path: Path):
    save_path = path.joinpath("plots_grid_heatmaps")
    os.makedirs(save_path, exist_ok=True)

    for use_case, buildings in itertools.product(
            ["hybrid", "adv_retrofit", "retrofit"],
            [True, False]
    ):
        plot_heat_map_trafo_size_with_uncertainty(
            path=path, save_path=save_path,
            oldbuildings=buildings, use_case=use_case,
        )


def plot_analysis_of_effects_with_uncertainty(
        path: Path,
        oldbuildings: bool = False
):
    if oldbuildings:
        grid_name = "oldbuildings"
    else:
        grid_name = "newbuildings"
    analysis_name = "Analyse"
    with open(path.joinpath(f"{grid_name}_max_data.json"), "r") as file:
        data = json.load(file)
    # Filter cases
    data = {k: v for k, v in data.items() if k.startswith(analysis_name)}
    analysis_order = {
        "HP": "heat_pump",
        "HR": "heating_rod",
        "EMobility": "e_mobility",
        "PV": "pv",
        "PVBat": "battery"
    }

    labels = []
    fixed_technologies = []
    technologies = []
    curves_max = []
    curves_mid = []
    curves_min = []
    curves_t_oda = []

    def _get_min_trafo_size_key(d: dict, metric: str):
        sizes = np.array([int(s.replace(" kVA", "")) for s in d.keys()])
        s_max = np.array([_max[metric] for _max in d.values()])
        load = s_max / sizes
        return f"{sizes[load < 1].min()} kVA"

    for analysis, tech_name in analysis_order.items():
        case_name = f"{analysis_name}{analysis}"
        fixed_technologies.append(tech_name)
        # data[case_name] = {"0%": {1000 kVa: {main_metric: 0, max_metric: 0}}}
        for percent_variation in [0, 20, 40, 60, 80, 100]:
            _percent_str = f"{percent_variation}%"
            trafo_data = data[case_name][_percent_str]
            labels.append(_percent_str)
            if percent_variation == 40:
                technologies.append(fixed_technologies.copy())
            else:
                technologies.append([])
            min_trafo_size = _get_min_trafo_size_key(trafo_data, MAX_MC_METRIC)
            curves_max.append(trafo_data[min_trafo_size][MAX_MC_METRIC])
            curves_mid.append(trafo_data[min_trafo_size][MAIN_MC_METRIC])
            curves_min.append(trafo_data[min_trafo_size][MIN_MC_METRIC])
            curves_t_oda.append(trafo_data[min_trafo_size]["t_oda"])
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=2, height_factor=1.5))
    ax_twin = ax.twinx()
    ax_twin.set_ylabel("$T_\mathrm{Oda}$ in °C")
    x_ticks = range(len(curves_mid))
    for i in range(int(len(curves_mid) / 6)):
        ax.axvline(5.5 + (i * 6), color="black")

    ax.plot(x_ticks, curves_mid, color=EBCColors.ebc_palette_sort_2[0])
    uncertainty_kwargs = dict(
        edgecolor=None, alpha=0.5, facecolor=EBCColors.ebc_palette_sort_2[0]
    )
    ax.fill_between(x_ticks, curves_min, curves_max, **uncertainty_kwargs)
    ax_twin.plot(x_ticks, curves_t_oda, color=EBCColors.ebc_palette_sort_2[1])
    ax.set_ylabel(METRIC_DATA["s_trafo"]["label"])
    ax.set_xticks(x_ticks)
    icon_plotting.add_images_to_axis(
        technologies=technologies,
        ax=ax,
        which_axis="x",
        width=0.1,
        distance_to_others=0.02,
        y_offset=0.05,
        x_offset=-0.025
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=90)

    set_color_of_axis(axis=ax.yaxis, color=EBCColors.ebc_palette_sort_2[0])
    set_color_of_axis(axis=ax_twin.yaxis, color=EBCColors.ebc_palette_sort_2[1])

    fig.tight_layout()
    fig.savefig(path.joinpath(f"{grid_name}_{analysis_name}_s_max.png"))


def copy_selection_for_paper(src, dst):
    files = [
        "oldbuildings_hybrid_main_Trafo-Size.png",
        "oldbuildings_hybrid_max_vm_pu_min smaller 0.95.png",
        "oldbuildings_hybrid_max_Trafo-Size.png",
        "oldbuildings_hybrid_max_max_line_loading_max.png",
        "newbuildings_hybrid_max_max_line_loading_max.png",
        "newbuildings_hybrid_max_Trafo-Size.png",
        "newbuildings_hybrid_max_vm_pu_min smaller 0.95.png",
        "newbuildings_hybrid_main_Trafo-Size.png",
        "oldbuildings_retrofit_main_Trafo-Size.png",
        "oldbuildings_adv_retrofit_main_Trafo-Size.png",
        "oldbuildings_retrofit_max_Trafo-Size.png",
        "oldbuildings_adv_retrofit_max_Trafo-Size.png",
    ]
    import shutil
    for file in files:
        shutil.copy2(src.joinpath("plots_grid_heatmaps", file), dst.joinpath(file))


if __name__ == '__main__':
    from hps_grid_interaction import RESULTS_MONTE_CARLO_FOLDER

    # Load RC Params
    PlotConfig.load_default()

    # aggregate_simultaneity_factors(path=PATH)
    generate_all_cases(RESULTS_MONTE_CARLO_FOLDER, with_plot=True, oldbuildings=True, use_mp=True)
    # generate_all_cases(PATH, with_plot=True, oldbuildings=False, use_mp=True)
    # plot_all_heat_map_trafo_size(PATH)
    # plot_analysis_of_effects_with_uncertainty(path=PATH, oldbuildings=False)
    # plot_analysis_of_effects_with_uncertainty(path=PATH, oldbuildings=True)
