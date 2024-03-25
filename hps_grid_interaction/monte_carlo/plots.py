import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData
import seaborn as sns

from hps_grid_interaction import DATA_PATH
from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.emissions import COLUMNS_EMISSIONS, get_emission_options
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size, icon_plotting

logger = logging.getLogger(__name__)


def get_label_and_factor(metric):
    if metric == "max":
        return "$P_\mathrm{el,max}$ in kW", 1
    elif metric == "sum":
        return "$W_\mathrm{el,Ges}$ in MWh", 1e-3
    elif metric == "value":
        return "$P_\mathrm{el}$ in kW", 1
    raise ValueError


def _get_grid_sum(grid_tsd):
    sum_grid = grid_tsd[0].copy().values
    for df in grid_tsd[1:]:
        sum_grid += df.values
    return sum_grid


def plot_quota_case_with_images(
        quota_variation: "QuotaVariation",
        ax: plt.axes,
        which_axis: str = None,
        width: float = 0.1,
        title_offset: float = 0.0,
        distance_to_others: float = 0.01
):
    icon_plotting.add_images_to_title(
        technologies=quota_variation.fixed_technologies, ax=ax,
        width=width, offset=title_offset,
        distance_to_others=distance_to_others
    )
    if which_axis is None:
        return ax
    if isinstance(quota_variation.varying_technologies, dict):
        technology, _ = quota_variation.get_single_varying_technology_name_and_quotas()
        icon_plotting.add_image_and_text_as_label(
            ax=ax, which_axis=which_axis, technology=technology, width=width,
            ticklabels=quota_variation.get_varying_technology_ids(),
            distance_to_others=distance_to_others
        )
    else:
        icon_plotting.add_images_to_axis(
            technologies=quota_variation.varying_technologies, ax=ax,
            which_axis=which_axis, width=width,
            distance_to_others=distance_to_others
        )
    return ax


def plot_time_series(quota_case_grid_data: dict, save_path, quota_variation: "QuotaVariation"):
    label = get_label_and_factor("value")[0]
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=get_figure_size(n_columns=2))
    fig_yearly, ax_yearly = plt.subplots(1, 1, figsize=get_figure_size(n_columns=2))

    #ax = [ax]
    t_oda = load_outdoor_air_temperature()
    bins = np.linspace(t_oda.values[1:-1, 0].min(), t_oda.values[1:-1, 0].max(), num=30)
    categories = pd.cut(t_oda.values[1:-1, 0], bins, labels=False)
    idx = 0
    quota_values = quota_variation.get_varying_technology_ids()
    for quota_case, grid_time_series_data in quota_case_grid_data.items():
        df_sum = _get_grid_sum(grid_time_series_data)
        min_curve = []
        max_curve = []
        for bin_idx in range(len(bins)):
            bin_mask = (categories == bin_idx)
            if np.any(bin_mask):
                max_curve.append(df_sum[bin_mask].max())
                min_curve.append(df_sum[bin_mask].min())
            else:
                max_curve.append(np.NAN)
                min_curve.append(np.NAN)

        ax[0].plot(np.arange(len(df_sum)) / 4, np.sort(df_sum)[::-1],
                   label=quota_values[idx])
        ax[1].plot(bins, max_curve, label=quota_values[idx], linestyle="-")
        ax_yearly.plot()
        ax_yearly.plot(np.arange(len(df_sum)) / 4, df_sum,
                       label=quota_values[idx])
        idx += 1
    if isinstance(quota_variation.varying_technologies, dict):
        tech_name = quota_variation.get_single_varying_technology_name_and_quotas()[0]
        fig.suptitle(f"Variation of {tech_name.capitalize()}-quota")

    ax[0].set_ylabel(label)
    ax[1].set_xlabel("$T_\mathrm{Oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[0])
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[1])
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"sorted_time_series_plot.png"), dpi=400)
    ax_yearly.set_ylabel(label)
    ax_yearly.set_xlabel("Hour of year in h")
    ax_yearly.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax_yearly, width=0.05)
    fig_yearly.tight_layout()
    fig_yearly.savefig(save_path.joinpath(f"annual_time_series_plot.png"), dpi=400)


def plot_results_all_cases(path: Path, point: str = "ONT", metric: str = "sum"):
    with open(path, "r") as file:
        results = json.load(file)
    y_label, fac = get_label_and_factor(metric)
    cases = {}
    emissions = {}
    for tech in ["Hybrid", "Monovalent"]:
        for case, case_values in results.items():
            cases[tech + "_" + case] = case_values["grid"][tech][metric][point] * fac
            values = case_values["emissions"][tech]
            for key in get_emission_options():
                emissions[tech + "_" + case + "_" + key] = values[key + "_gas"] + values[key + "_electricity"]
    cases = dict(sorted(cases.items(), key=lambda item: item[1]))
    emissions = dict(sorted(emissions.items(), key=lambda item: item[1]))

    def _plot_barh(_data, _y_label):
        fig, ax = plt.subplots(figsize=[9.3, 7.5 * len(_data) / 11])
        points = list(_data.keys())
        x_pos = np.arange(len(points))
        bar_args = dict(align='center', ecolor='black', height=0.4)
        ax.barh(x_pos, list(_data.values()), color="red", **bar_args)
        ax.set_xlabel(_y_label)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(points)
        ax.xaxis.grid(True)
        fig.tight_layout()
        return fig

    fig = _plot_barh(cases, y_label)
    fig.savefig(path.parent.joinpath(f"results_{metric}.png"))
    fig = _plot_barh(emissions, "CO2-Emissionsn in kg")
    fig.savefig(path.parent.joinpath(f"emissions.png"))


def plot_monte_carlo_violin(
        data: dict,
        metric: str,
        save_path: Path,
        quota_variation: "QuotaVariation",
        points: list = None
):
    data = data[metric]
    if points is None:
        points = ["ONT"]
    n_subplots = len(quota_variation.quota_cases)
    label, factor = get_label_and_factor(metric)
    for point in points:
        values = data[point]
        fig, axes = plt.subplots(figsize=get_figure_size(n_columns=1))
        data_dict = {}
        for varying_tech, quota_case in zip(
                quota_variation.get_varying_technology_ids(),
                quota_variation.quota_cases.keys()
        ):
            data_dict[varying_tech] = np.array(values[quota_case]) * factor
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data_dict.items()]))
        sns.violinplot(data=df, ax=axes, orient='h')
        axes.set_xlabel(label)
        axes.set_yticks(list(range(len(quota_variation.quota_cases))))
        plot_quota_case_with_images(quota_variation=quota_variation, ax=axes, which_axis="y")
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"monte_carlo_violin_{point}_{metric}.png"), dpi=400)
    plt.close("all")


def plot_monte_carlo_bars(
        data: dict,
        metric: str,
        save_path: Path,
        quota_variation: "QuotaVariation",
        points: list = None
):
    n_bars = 1
    if points is None:
        points = ["ONT"]
    max_data = data[metric]
    y_label, factor = get_label_and_factor(metric)
    # Violins
    plt.figure()
    plot_data = {point: {"mean": [], "std": []} for point in points}
    for point in points:
        data = max_data[point]
        for quota_case in quota_variation.quota_cases:
            plot_data[point]["mean"].append(np.mean(data[quota_case]) * factor)
            plot_data[point]["std"].append(np.std(data[quota_case]) * factor)
    fig, axes = plt.subplots(len(points), 1, figsize=get_figure_size(n_columns=1))
    if len(points) == 1:
        axes = [axes]
    x_pos = np.arange(len(quota_variation.quota_cases))
    bar_width = 0.8 / n_bars
    bar_args = dict(align='center', ecolor='black', width=bar_width)
    idx = 0
    for ax, point in zip(axes, points):
        ax.bar(
            x_pos - 0.4 + (1 / 2 + idx) * bar_width, plot_data[point]["mean"],
            yerr=plot_data[point]["std"],
            color=EBCColors.ebc_palette_sort_2[idx],
            **bar_args,
        )
        ax.set_ylabel(point + " " + y_label)
        ax.set_xticks(x_pos)
        plot_quota_case_with_images(ax=ax, quota_variation=quota_variation, which_axis="x")
        ax.yaxis.grid(True)

    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_{metric}.png"), dpi=400)
    plt.close("all")


def plot_cop_motivation():
    plt.rcParams.update({"figure.figsize": [6.24 * 1.2, 5.78 / 1.3]})
    tsd35 = TimeSeriesData(DATA_PATH.joinpath("GetCOPCurve308.mat")).to_df()
    tsd70 = TimeSeriesData(DATA_PATH.joinpath("GetCOPCurve343.mat")).to_df()
    plt.plot(tsd35.loc[1:, "TOda"] - 273.15, tsd35.loc[1:, "sigBusGen.COP"], label="FBH 35 °C")
    plt.plot(tsd70.loc[1:, "TOda"] - 273.15, tsd70.loc[1:, "sigBusGen.COP"], label="Radiator 70 °C")
    plt.legend()
    plt.ylabel("$COP$")
    plt.xlabel("Außentemperatur in °C")
    plt.tight_layout()
    plt.savefig("plots/COP Motivation.png")
    plt.show()

