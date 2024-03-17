import json
import logging
import pathlib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData
import seaborn as sns

from hps_grid_interaction import DATA_PATH
from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.emissions import COLUMNS_EMISSIONS, get_emission_options
from hps_grid_interaction.plotting.config import EBCColors

logger = logging.getLogger(__name__)


def _get_grid_sum(grid_tsd):
    sum_grid = grid_tsd[0].copy().values
    for df in grid_tsd[1:]:
        sum_grid += df.values
    return sum_grid


def is_number(s: str):
    try:
        _ = float(s)
        return True
    except ValueError:
        return False


def _get_quota_value(quota_case, in_percent):
    quota_case_value = quota_case.split("_")[-1]
    if in_percent:
        if is_number(quota_case_value):
            quota_case_value += " %"
        if len(quota_case_value) > 15:
            return quota_case_value[len(quota_case_value)-15:]
        else:
            return quota_case_value
    if not is_number(quota_case_value):
        raise ValueError(f"Given quota_case is not a number: {quota_case_value}")
    return int(quota_case.split("_")[-1])


def _get_quota_name_except_value(quota_case):
    quota_case = _alter_quota_case(quota_case)
    if "with" in quota_case:
        first, second = quota_case.split("_with_")
        return "-".join(first.split("_")[:-1] + second.split("_")[:-1])
    return "-".join(quota_case.split("_")[:-2])


def _alter_quota_case(quota_case):
    quota_case = quota_case.replace("av_", "Average_")
    quota_case = quota_case.replace("pv_bat_", "PV+Battery_")
    quota_case = quota_case.replace("pv_", "PV_")
    quota_case = quota_case.replace("hyb_", "Hybrid_")
    quota_case = quota_case.replace("e_mob", "EMobility")
    return quota_case


def _get_quota_varying_name(quota_case):
    quota_case = _alter_quota_case(quota_case)

    if "with" in quota_case:
        first, second = quota_case.split("_with_")
        return first.split("_")[-1]
    return quota_case.split("_")[-2]


def plot_quota_case_with_images(x_ticks: list, quota_cases: list, ax: plt.axes, image_width: float = 0.4):
    from matplotlib.image import BboxImage, imread
    from matplotlib.transforms import Bbox
    # remove tick labels
    ax.set_xticklabels([])
    for x_tick, quota_case in zip(x_ticks, quota_cases):
        technologies = _get_technologies(quota_case)
        for idx, technology in enumerate(technologies):
            tick_y_position = -0.2 - image_width * (1 + idx)

            lower_corner = ax.transData.transform((x_tick - image_width / 2, tick_y_position - image_width / 2))
            upper_corner = ax.transData.transform((x_tick + image_width / 2, tick_y_position + image_width / 2))

            bbox_image = BboxImage(
                Bbox([lower_corner[0], lower_corner[1], upper_corner[0], upper_corner[1], ]),
                norm=None,
                origin=None,
                clip_on=False,
            )

            bbox_image.set_data(imread(DATA_PATH.joinpath("icons", f"{technology}.png")))
            ax.add_artist(bbox_image)
    return ax


def plot_time_series(quota_case_grid_data: dict, save_path):
    label = get_label_and_factor("value")[0]
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig_yearly, ax_yearly = plt.subplots(1, 1, figsize=(27, 9))

    #ax = [ax]
    t_oda = load_outdoor_air_temperature()
    bins = np.linspace(t_oda.values[1:-1, 0].min(), t_oda.values[1:-1, 0].max(), num=30)
    categories = pd.cut(t_oda.values[1:-1, 0], bins, labels=False)
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
                   label=_get_quota_value(quota_case, True))
        #ax[1].plot(bins, min_curve, label=_get_quota_value(quota_case, True), linestyle="--")
        ax[1].plot(bins, max_curve, label=_get_quota_value(quota_case, True), linestyle="-")
        ax_yearly.plot()
        ax_yearly.plot(np.arange(len(df_sum)) / 4, df_sum,
                       label=_get_quota_value(quota_case, True))
    fig.suptitle(_get_quota_name_except_value(quota_case))
    ax[0].set_ylabel(label)
    ax[1].set_xlabel("$T_\mathrm{Oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    #ax[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"sorted_time_series_plot.png"))
    ax_yearly.set_ylabel(label)
    ax_yearly.set_xlabel("Hour of year in h")
    ax_yearly.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    fig_yearly.tight_layout()
    fig_yearly.savefig(save_path.joinpath(f"annual_time_series_plot.png"))


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


def plot_monte_carlo_violin(data: dict, metric: str, save_path: Path, quota_cases: dict, points: list = None):
    data = data[metric]
    if points is None:
        points = ["ONT"]
    n_subplots = len(quota_cases)
    #if n_subplots > 8:
    #    logger.error("Won't plot violins, too many quota_cases: %s", n_subplots)
    label, factor = get_label_and_factor(metric)
    for point in points:
        values = data[point]
        fig, axes = plt.subplots()
        df = pd.DataFrame(
            {_get_quota_value(quota_case, True): np.array(values[quota_case]) * factor for quota_case in quota_cases})
        with sns.plotting_context(rc={"font.size": 14}):
            sns.set(style="whitegrid")
            sns.violinplot(data=df, ax=axes, orient='h')
        axes.set_xlabel(label)
        axes.set_ylabel(_get_quota_varying_name(list(quota_cases.keys())[0]))
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"monte_carlo_violin_{point}_{metric}.png"))
    plt.close("all")


def get_label_and_factor(metric):
    if metric == "max":
        return "$P_\mathrm{el,max}$ in kW", 1
    elif metric == "sum":
        return "$W_\mathrm{el,Ges}$ in MWh", 1e-3
    elif metric == "value":
        return "$P_\mathrm{el}$ in kW", 1
    raise ValueError


def plot_monte_carlo_bars(data: dict, metric: str, save_path: Path, quota_cases: dict, points: list = None):
    n_bars = 1
    if n_bars > 5:
        logger.error("Won't plot monte carlo bars, too many different quota_cases: %s", n_bars)
        return
    if points is None:
        points = ["ONT"]
    max_data = data[metric]
    y_label, factor = get_label_and_factor(metric)
    # Violins
    plt.figure()
    plot_data = {point: {"mean": [], "std": []} for point in points}
    for point in points:
        data = max_data[point]
        for quota_case in quota_cases:
            plot_data[point]["mean"].append(np.mean(data[quota_case]) * factor)
            plot_data[point]["std"].append(np.std(data[quota_case]) * factor)
    fig, axes = plt.subplots(len(points), 1)
    if len(points) == 1:
        axes = [axes]
    x_pos = np.arange(len(quota_cases))
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
        # ax.legend()
        ax.set_xticklabels([_get_quota_value(quota_case, True) for quota_case in quota_cases.keys()], rotation=90)
        ax.set_xlabel(_get_quota_varying_name(list(quota_cases.keys())[0]))
        ax.yaxis.grid(True)
    fig.suptitle(_get_quota_name_except_value(list(quota_cases.keys())[0]))

    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_{metric}.png"))
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


def plot_single_draw_scatter(hybrid_grid, monovalent_grid):
    label = "$P_\mathrm{el}$ in kW"

    t_oda = load_outdoor_air_temperature()
    scatter_kwargs = dict(s=2, marker="o")
    for name, hybrid, monovalent in zip(hybrid_grid.keys(), hybrid_grid.values(), monovalent_grid.values()):
        fig, ax = plt.subplots(1, 1, sharey=True)
        ax.set_ylabel(label)
        ax.set_xlabel("$T_\mathrm{Oda}$ in °C")
        ax.scatter(t_oda, monovalent, color="red", **scatter_kwargs, label="Monovalent")
        ax.scatter(t_oda, hybrid, color="blue", **scatter_kwargs, label="Hybrid")
        ax.legend()
        fig.suptitle(name)
        fig.savefig(f"plots/Scatter_results_{name}.png")
        plt.close(fig)
