import json
import logging
import pathlib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData

from hps_grid_interaction.utils import load_outdoor_air_temperature
from hps_grid_interaction.emissions import COLUMNS_EMISSIONS, get_emission_options

logger = logging.getLogger(__name__)


def plot_time_series(data, quota_case, metric, save_path, arg):
    plt.figure()
    plt.suptitle(quota_case)
    for point in data["tsd_data"].keys():
        tsd = data["tsd_data"][point][quota_case][arg]
        plt.plot(tsd, label=point)
        plt.ylabel(get_label_and_factor("value")[0])
    plt.legend(loc="upper right", ncol=2)
    plt.savefig(save_path.joinpath(f"time_series_plot_{quota_case}_{metric}.png"))


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


def plot_single_draw_max(hybrid_grid, monovalent_grid):
    # Maximal value
    max_data = {}
    for point, hybrid, monovalent in zip(hybrid_grid.keys(), hybrid_grid.values(), monovalent_grid.values()):
        max_data[point] = {"Hybrid": hybrid.max(), "Monovalent": monovalent.max()}
    df = pd.DataFrame(max_data).transpose()

    label = "$P_\mathrm{el}$ in kW"
    df.plot.bar()
    plt.ylabel(label)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/Max_results.png")


def plot_monte_carlo_violin(data: dict, save_path: Path, quota_cases: dict, points: list = None):
    if points is None:
        points = ["ONT"]
    n_subplots = len(quota_cases)
    if n_subplots > 8:
        logger.error("Won't plot violins, too many quota_cases: %s", n_subplots)
    y_label, factor = get_label_and_factor(metric)
    # Violins
    plt.figure()
    for point in points:
        values = data[point]
        fig, ax = plt.subplots(1, n_subplots, sharey=True)
        if n_subplots == 1:
            ax = [ax]
        for _ax, label in zip(ax, quota_cases.keys()):
            data = values[label]
            violin_settings = dict(points=100, showextrema=True)
            _ax.violinplot(np.array(data) * factor, **violin_settings)
            _ax.set_xticks(np.arange(1, len([label]) + 1), labels=[label])
        ax[0].set_ylabel(y_label)
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
    from hps_grid_interaction.plotting.config import EBCColors
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
        #ax.legend()
        ax.set_xticklabels(list(quota_cases.keys()), rotation=90)
        ax.yaxis.grid(True)

    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_{metric}.png"))
    plt.close("all")


def plot_cop_motivation():
    path = pathlib.Path(__file__).absolute().parents[1].joinpath("data")
    plt.rcParams.update({"figure.figsize": [6.24 * 1.2, 5.78 / 1.3]})
    tsd35 = TimeSeriesData(path.joinpath("GetCOPCurve308.mat")).to_df()
    tsd70 = TimeSeriesData(path.joinpath("GetCOPCurve343.mat")).to_df()
    plt.plot(tsd35.loc[1:, "TOda"] - 273.15, tsd35.loc[1:, "sigBusGen.COP"], label="FBH 35 °C")
    plt.plot(tsd70.loc[1:, "TOda"] - 273.15, tsd70.loc[1:, "sigBusGen.COP"], label="Radiator 70 °C")
    plt.legend()
    plt.ylabel("$COP$")
    plt.xlabel("Außentemperatur in °C")
    plt.tight_layout()
    plt.savefig("plots/COP Motivation.png")
    plt.show()


def plot_single_draw_violin(hybrid_grid, monovalent_grid):
    label = "$P_\mathrm{el}$ in kW"
    # Violins
    plt.figure()
    fig, ax = plt.subplots(2, 1, sharex=True)
    position_labels = list(hybrid_grid.keys())
    violin_settings = dict(points=100, showextrema=True)
    ax[0].violinplot(hybrid_grid.values(), **violin_settings)
    ax[0].set_xticks(np.arange(1, len(position_labels) + 1), labels=position_labels)
    ax[1].violinplot(monovalent_grid.values(), **violin_settings)
    ax[1].set_xticks(np.arange(1, len(position_labels) + 1), labels=position_labels)
    ax[1].set_ylabel(label)
    ax[0].set_ylabel(label)
    fig.tight_layout()


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
