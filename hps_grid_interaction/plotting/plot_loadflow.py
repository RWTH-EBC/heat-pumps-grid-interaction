import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
    "s_trafo": {"label": "$S$ in kVA", "opt": "max", "label_abs": "$|S|$ in kVA"},
    "vm_pu_min": {"label": "$V_\mathrm{min}$ in p.u.", "opt": "min",
                  "axhlines": [0.9, 0.95, 0.97], "min_max": {"vmin": 0.9, "vmax": 1}},
#    "vm_pu_max": {"label": "$V_\mathrm{max}$ in p.u.", "opt": "max", "min_max": {"vmin": 0.85, "vmax": 1}},
    "max_line_loading": {"label": "$L_\mathrm{max}$ in %", "opt": "max", "min_max": {"vmin": 0, "vmax": 100}},
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


def load_case_and_trafo_data(path: Path, quota_variation: QuotaVariation, grid: str, monte_carlo_metrics: list):
    with open(path.joinpath("results_to_plot.json"), "r") as file:
        results = json.load(file)
    case_and_transformer_data = {}
    name_value_dict = quota_variation.get_quota_case_name_and_value_dict()
    for monte_carlo_metric in monte_carlo_metrics:
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
        main_metric: str = "argmean",
        max_metric: str = "argmax",
        min_metric: str = "argmin"
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
            fig_title = f"{varying_tech_name}-quota variation | " + fig_title
        save_name = f"trafo={fixed_trafo_size}"

    fig, ax = plt.subplots(1, 2, sharey=False, figsize=get_figure_size(n_columns=2))
    t_oda = load_outdoor_air_temperature()
    bins = np.linspace(t_oda.values[1:-1, 0].min(), t_oda.values[1:-1, 0].max(), num=30)
    categories = pd.cut(t_oda.values[1:-1, 0], bins, labels=False)
    idx_case = 0
    for _label, _monte_carlo_tsd in _data.items():
        curves = {}
        for monte_carlo_metric, time_series_data in _monte_carlo_tsd.items():
            curve = []
            for bin_idx in range(len(bins)):
                bin_mask = (categories == bin_idx)
                if np.any(bin_mask):
                    if metric_data[metric]["opt"] == "min":
                        curve.append(np.abs(time_series_data[bin_mask]).min())
                    else:
                        curve.append(np.abs(time_series_data[bin_mask]).max())
                else:
                    curve.append(np.NAN)
            curves[monte_carlo_metric] = curve
        main_tsd = _monte_carlo_tsd[main_metric]
        max_tsd = _monte_carlo_tsd[max_metric]
        min_tsd = _monte_carlo_tsd[min_metric]
        all_curves = np.array([curves[min_metric], curves[max_metric], curves[main_metric]])
        main_cluster = curves[main_metric]
        max_cluster = np.max(all_curves, axis=1)
        min_cluster = np.min(all_curves, axis=1)
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

    axhlines = metric_data[metric].get("axhlines", None)
    if axhlines is not None:
        for hline in axhlines:
            for _ax in ax:
                _ax.axhline(hline, color="black")
    fig.suptitle(fig_title)
    y_label_non_abs = metric_data[metric]["label"]
    ax[0].set_ylabel(y_label_non_abs)
    ax[1].set_ylabel(metric_data[metric].get("label_abs", y_label_non_abs))
    ax[1].set_xlabel("$T_\mathrm{oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[0])
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[1])

    fig.suptitle(fig_title)
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path.joinpath(f"{metric}_{save_name}.png"))
    plt.close("all")


def get_statistics(case_and_trafo_data: dict, path: Path):
    df = pd.DataFrame()
    idx = 0
    for case, trafo_data in case_and_trafo_data.items():
        for trafo_size, trafo_results in trafo_data.items():
            for monte_carlo_metric, trafo_results_monte_carlo in trafo_results.items():
                df.loc[idx, "Trafo-Size"] = trafo_size
                df.loc[idx, "case"] = case
                df.loc[idx, "Monte-Carlo Metric"] = monte_carlo_metric
                for metric, settings in metric_data.items():
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
        quota_variation: QuotaVariation,
        main_metric: str = "argmean",
        max_metric: str = "argmax",
        min_metric: str = "argmin"
):
    cases = list(case_and_trafo_data.keys())
    trafo_sizes = list(case_and_trafo_data[cases[0]])
    save_path = path.joinpath("plots")
    os.makedirs(save_path, exist_ok=True)
    for metric in metric_data.keys():
        only_one_trafo_is_enough = metric in ["p_trafo", "s_trafo", "q_trafo"]
        #only_one_trafo_is_enough = False
        kwargs = dict(
            main_metric=main_metric,
            max_metric=max_metric,
            min_metric=min_metric,
            case_and_trafo_data=case_and_trafo_data,
            metric=metric,
            save_path=save_path,
            quota_variation=quota_variation
        )
        for trafo_size in trafo_sizes:
            plot_time_series(fixed_trafo_size=trafo_size, **kwargs)
            if only_one_trafo_is_enough:
                break
        for case in cases:
            plot_time_series(fixed_case=case, **kwargs)


def generate_all_cases(
        path: Path, with_plot: bool, oldbuildings=True, use_mp: bool = True,
        main_metric: str = "argmean",
        max_metric: str = "argpercentile_95",
        min_metric: str = "argpercentile_5"
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
        if folder.startswith(grid_case) and os.path.isdir(path.joinpath(folder))
    ]
    kwargs_mp = []
    for folder in folders:
        case = folder.replace(grid_case, "")
        kwargs_mp.append(dict(
            case_path=path.joinpath(folder),
            case=case,
            quota_variation=all_quota_cases[case],
            grid_str=grid_str, with_plot=with_plot,
            main_metric=main_metric,
            max_metric=max_metric,
            min_metric=min_metric,
        ))

    idx = 0
    if use_mp:
        import multiprocessing as mp
        pool = mp.Pool(processes=30)

        for df in pool.imap_unordered(create_plots_and_get_df, kwargs_mp):
            dfs.append(df)
            print(f"Ran {idx + 1}/{len(folders)} folders")
            idx += 1
    else:
        for kwargs in kwargs_mp:
            dfs.append(create_plots_and_get_df(kwargs))
            print(f"Ran {idx + 1}/{len(folders)} folders")
            idx += 1
    dfs = pd.concat(dfs)
    dfs.index = range(len(dfs))
    dfs.to_excel(path.joinpath(f"{grid_case}all_grid_results_ex.xlsx"))


def create_plots_and_get_df(kwargs):
    # Load RC Params
    PlotConfig.load_default()

    quota_variation = kwargs["quota_variation"]
    grid_str = kwargs["grid_str"]
    case_path = kwargs["case_path"]
    case = kwargs["case"]
    with_plot = kwargs.get("with_plot", True)
    main_metric = kwargs["main_metric"]
    max_metric = kwargs["max_metric"]
    min_metric = kwargs["min_metric"]

    print(f"Extracting and plotting {case_path}")
    try:
        case_and_trafo_data = load_case_and_trafo_data(case_path, quota_variation=quota_variation, grid=grid_str,
                                                       monte_carlo_metrics=[main_metric, max_metric, min_metric])
        df = get_statistics(case_and_trafo_data=case_and_trafo_data, path=case_path)
        df.loc[:, "quota_cases"] = case
        if with_plot:
            plot_all_metrics_and_trafos(
                case_and_trafo_data, case_path, quota_variation,
                main_metric=main_metric, max_metric=max_metric, min_metric=min_metric
            )
            for second_metric in calculated_metrics.keys():
                plot_required_trafo_size(
                    path=case_path.joinpath("plots"),
                    df=df,
                    quota_variation=quota_variation,
                    design_metric="percent_max_trafo_load in %",
                    design_value=100,
                    second_metric=second_metric,
                    monte_carlo_metric=max_metric
                )
            plot_grid_as_heatmap(case_and_trafo_data, case_path, monte_carlo_metric=main_metric)

        print(f"Extracted and plotted {case_path}")
        return df
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
        second_metric: str = "vm_pu_min smaller 0.97",
        monte_carlo_metric: str = "argmax"
):
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
    from copy import deepcopy
    quota_variation_copy = deepcopy(quota_variation)
    if isinstance(quota_variation.varying_technologies, dict):
        name, values = quota_variation.get_single_varying_technology_name_and_quotas()
        quota_variation_copy.varying_technologies = {name: values[::-1]}
    else:
        quota_variation_copy.varying_technologies = quota_variation_copy.varying_technologies[::-1]

    plot_quota_case_with_images(quota_variation=quota_variation_copy, ax=ax, which_axis="y", title_offset=0.2)
    #ax.set_yticklabels(df_plot.index)
    set_color_of_axis(axis=ax.xaxis, color=EBCColors.ebc_palette_sort_2[0])

    fig.tight_layout(pad=0.15, w_pad=0.1, h_pad=0.1)
    fig.savefig(path.joinpath(f"MinTrafoDesign_{second_metric}.png"))


def plot_grid_as_heatmap(case_and_trafo_data: dict, save_path: Path, monte_carlo_metric: str = "argmean"):
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
        fig, ax = plt.subplots(n_cases, n_trafo_sizes, figsize=get_figure_size(n_columns=1 * n_trafo_sizes, height_factor=0.7 * n_cases), sharex=True, sharey=True)
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
                if metric_name not in metric_data:
                    continue
                metric_kwargs = metric_data.get(metric_name, {})
                if metric_kwargs["opt"] == "min":
                    cmap = "flare_r"
                else:
                    cmap = "flare"
                ax[idx_case, idx_trafo] = plot_heat_map_on_grid_image(
                    ax=ax[idx_case, idx_trafo], df=df,
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
        fig.suptitle(metric_data[metric_name].get("label", metric_name))
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"{metric}.png"))
        plt.close("all")


def plot_heat_map_on_grid_image(ax: plt.axes, cmap, df: pd.DataFrame, heatmap_kwargs: dict = None):
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    sns.heatmap(df.astype(float), ax=ax, cmap=cmap, linewidths=0,
                zorder=1, linecolor='black', **heatmap_kwargs)
    ax.imshow(
        plt.imread(DATA_PATH.joinpath("oldbuildings_grid.png"), format="png"),
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


if __name__ == '__main__':
    from hps_grid_interaction import RESULTS_MONTE_CARLO_FOLDER
    PATH = RESULTS_MONTE_CARLO_FOLDER
    PATH = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\03_monte_carlo")
    #aggregate_simultaneity_factors(path=PATH)
    #generate_all_cases(PATH, with_plot=False, oldbuildings=True, use_mp=True)
    generate_all_cases(PATH, with_plot=True, oldbuildings=False, use_mp=False)
