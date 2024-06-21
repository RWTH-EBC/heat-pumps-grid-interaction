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
from hps_grid_interaction.plotting.config import EBCColors
from hps_grid_interaction.plotting import get_figure_size, icon_plotting

logger = logging.getLogger(__name__)


def get_label_and_factor(metric):
    if metric == "max":
        return "$P_\mathrm{el,max}$ in kW", 1
    elif metric == "sum":
        return "$E_\mathrm{el,tot}$ in MWh", 1e-3
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

    # ax = [ax]
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
    ax[1].set_xlabel("$T_\mathrm{oda}$ in °C")
    ax[0].set_xlabel("Hours in year")
    ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[0])
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax[1])
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"sorted_time_series_plot.png"))
    ax_yearly.set_ylabel(label)
    ax_yearly.set_xlabel("Hour of year in h")
    ax_yearly.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
    plot_quota_case_with_images(quota_variation=quota_variation, ax=ax_yearly, width=0.05)
    fig_yearly.tight_layout()
    fig_yearly.savefig(save_path.joinpath(f"annual_time_series_plot.png"))


def plot_monte_carlo_violin(
        data: dict,
        metric: str,
        save_path: Path,
        quota_variation: "QuotaVariation",
        points: list = None
):
    data = data[metric]
    if points is None:
        points = ["Trafo"]
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
        sns.violinplot(
            data=df, ax=axes, orient='h',
            palette=EBCColors.ebc_palette_sort_2[:len(quota_variation.quota_cases)])
        axes.set_xlabel(label)
        axes.set_yticks(list(range(len(quota_variation.quota_cases))))
        plot_quota_case_with_images(quota_variation=quota_variation, ax=axes, which_axis="y")
        fig.tight_layout()
        fig.savefig(save_path.joinpath(f"monte_carlo_violin_{point}_{metric}.png"))
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
        points = ["Trafo"]
    max_data = data[metric]
    y_label, factor = get_label_and_factor(metric)
    # Violins
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
        for quota_idx, _x_pos in enumerate(x_pos):
            ax.bar(
                _x_pos - 0.4 + (1 / 2 + idx) * bar_width,
                plot_data[point]["mean"][quota_idx],
                yerr=plot_data[point]["std"][quota_idx],
                color=EBCColors.ebc_palette_sort_2[quota_idx],
                **bar_args,
            )
        ax.set_ylabel(y_label)
        ax.set_xticks(x_pos)
        plot_quota_case_with_images(ax=ax, quota_variation=quota_variation, which_axis="x")
        ax.yaxis.grid(True)
        idx += 1

    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_{metric}.png"))
    plt.close("all")


def plot_monte_carlo_convergence(
        data: dict,
        metric: str,
        save_path: Path,
        quota_variation: "QuotaVariation"
):
    data = data[metric]["Trafo"]
    y_label, factor = get_label_and_factor(metric)
    # Violins
    n_rows = len(quota_variation.quota_cases)
    fig, axes = plt.subplots(n_rows, 1, sharex=True,
                             figsize=get_figure_size(n_columns=1.5, height_factor=0.8 * n_rows))
    mc_metrics = {
        "mean": (np.mean, []),
        "min": (np.min, []),
        "max": (np.max, []),
        "5p": (np.percentile, [5]),
        "95p": (np.percentile, [95]),
        "0.3p": (np.percentile, [0.3]),
        "99.7p": (np.percentile, [99.7]),
    }
    for quota_case, ax, quota_name in zip(
            quota_variation.quota_cases, axes, quota_variation.get_varying_technology_ids()
    ):
        quota_data = {mc_metric: [] for mc_metric in mc_metrics}
        n_monte_carlos = list(range(10, len(data[quota_case]), 10))
        for n_monte_carlo in n_monte_carlos:
            for mc_metric, func_args in mc_metrics.items():
                func, args = func_args
                quota_data[mc_metric].append(func(data[quota_case][:n_monte_carlo], *args) * factor)
        for key, values in quota_data.items():
            ax.plot(n_monte_carlos, values, label=key)
        ax.set_ylabel(y_label)
        ax.set_title(quota_name)

    axes[-1].set_xlabel("Monte-Carlo Iterations in -")
    axes[0].legend(bbox_to_anchor=(0, 1.1), ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_convergence_{metric}.png"))


def plot_cop_motivation():
    from hps_grid_interaction.bes_simulation import weather
    t_oda_year = weather.WeatherConfig().get_hourly_weather_data()
    t_oda_year = t_oda_year["t"].sort_values()
    t_oda_year[t_oda_year > 20] = 20  # No demand in summer
    min_t_oda = t_oda_year.min()

    bivalence_point = 1
    timestamp = t_oda_year[t_oda_year > bivalence_point].index[0]
    index_biv = t_oda_year.index.get_loc(timestamp)

    plt.rcParams.update({"figure.figsize": [6.24 * 1.2, 5.78 / 1.3], "font.size": 11})
    tsd35 = TimeSeriesData(DATA_PATH.joinpath("GetCOPCurve308.mat")).to_df()
    tsd70 = TimeSeriesData(DATA_PATH.joinpath("GetCOPCurve343.mat")).to_df()
    t_oda = tsd35.loc[:, "TOda"] - 273.15
    mask = t_oda > min_t_oda
    t_oda = t_oda[mask]

    # plt.figure()
    # plt.plot(t_oda, tsd35.loc[mask, "sigBusGen.COP"], label="FBH 35 °C")
    # plt.plot(t_oda, tsd70.loc[mask, "sigBusGen.COP"], label="Radiator 70 °C")
    # plt.legend()
    # plt.ylabel("$COP$")
    # plt.xlabel("Außentemperatur in °C")
    # plt.tight_layout()
    # plt.savefig("COP Motivation.png")
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="red", linestyle="-", label="Without retrofit"),
        Line2D([0], [0], color="green", linestyle="--", label="Retrofit"),
        Line2D([0], [0], color="black", linestyle="--", label="Hybrid heat pump"),
        Line2D([0], [0], color='w', label='Peak load', markerfacecolor='gray', markersize=15, marker="s"),
    ]

    Q_dem = 200 * (20 - t_oda)
    average_COP = (tsd35.loc[mask, "sigBusGen.COP"] + tsd70.loc[mask, "sigBusGen.COP"]) / 2
    P_el = Q_dem / average_COP
    P_el_retrofit = 0.5 * P_el
    mask_bivalent = t_oda > bivalence_point
    fig, ax = plt.subplots(
        3, 1,
        sharex=True, figsize=get_figure_size(n_columns=1, height_factor=1.7),
        gridspec_kw={'height_ratios': [1, 2, 2]}
    )
    ax[0].set_ylabel("Cumulative\nProbability")
    ax[0].plot(t_oda, t_oda.apply(lambda x: np.count_nonzero(t_oda_year <= x) / 8760).values, color="black")
    ax[0].set_yticks([0, 1])
    ax[1].plot(t_oda, Q_dem, color="red")
    ax[1].plot(t_oda, Q_dem, color="black", linestyle="--")
    ax[1].plot(t_oda, Q_dem * 0.5, color="green", linestyle="--")
    ax[1].set_yticks([0, Q_dem.values[0]])
    ax[1].set_yticklabels(["min", "max"])
    ax[1].set_ylabel("Heat demand")
    ax[2].plot(t_oda, P_el, color="red")
    ax[2].plot(t_oda[mask_bivalent], P_el[mask_bivalent], color="black", linestyle="--")
    ax[2].plot(t_oda, P_el_retrofit, color="green", linestyle="--")
    # Scatter
    scatter_kwargs = dict(s=15, marker="s")
    ax[2].scatter(t_oda.values[0], P_el.values[0], color="red", **scatter_kwargs)
    ax[2].scatter(t_oda[mask_bivalent].values[0], P_el[mask_bivalent].values[0], color="black", **scatter_kwargs)
    ax[2].scatter(t_oda.values[0], P_el_retrofit.values[0], color="green", **scatter_kwargs)
    ax[2].set_yticks([0, P_el.values[0]])
    ax[2].set_yticklabels(["min", "max"])
    ax[2].set_ylabel("Electricity load")
    ax[2].set_xlabel("Outdoor air temperature in °C")
    ax[2].set_xticks([-12, 20])
    ax[0].legend(handles=custom_lines, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)
    fig.tight_layout()
    fig.savefig(DATA_PATH.joinpath("Motivation_plot_oda.png"))

    Q_dem = 200 * (20 - t_oda_year)
    COP_t_oda = pd.Series(dict(zip(t_oda.values, average_COP.values)))
    COP_over_year = t_oda_year.apply(
        lambda x: COP_t_oda.reindex(COP_t_oda.index.union([x])).sort_index().interpolate().loc[x])
    P_el = Q_dem / COP_over_year
    P_el_retrofit = 0.5 * P_el
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=get_figure_size(n_columns=1, height_factor=1.4))
    ax[0].plot(range(len(Q_dem)), Q_dem, color="red")
    ax[0].plot(range(len(Q_dem)), Q_dem, color="black", linestyle="--")
    ax[0].plot(range(len(Q_dem)), Q_dem * 0.5, color="green", linestyle="--")
    ax[0].set_yticks([0, Q_dem.values[0]])
    ax[0].set_yticklabels(["min", "max"])
    ax[0].set_ylabel("Heat demand")
    ax[1].plot(range(len(Q_dem)), P_el, color="red")
    ax[1].plot(range(index_biv, len(Q_dem)), P_el[index_biv:], color="black", linestyle="--")
    ax[1].plot(range(len(Q_dem)), P_el_retrofit, color="green", linestyle="--")
    # Index 0 is Nan
    ax[1].scatter([0], P_el.values[1], color="red", **scatter_kwargs)
    ax[1].scatter(index_biv, P_el[index_biv:].values[0], color="black", **scatter_kwargs)
    ax[1].scatter([0], P_el_retrofit.values[1], color="green", **scatter_kwargs)
    ax[1].set_yticks([0, P_el.values[1]])
    ax[1].set_yticklabels(["min", "max"])
    ax[1].set_ylabel("Electricity load")
    ax[1].set_xlabel("Hours in year")
    ax[1].set_xticks([0, 8760])
    ax[0].legend(handles=custom_lines, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)
    fig.tight_layout()
    fig.savefig(DATA_PATH.joinpath("Motivation_plot_hours.png"))
    plt.show()


def plot_technology_choices_in_grid(df_grid: pd.DataFrame, choices_for_grid: dict, save_path: Path):
    df_grid_with_choices = df_grid.copy()
    choice_types = ["heat_supply_choice", "electricity_system_choice", "construction_type_choice"]
    for house_idx in df_grid_with_choices.index:
        for choice_type in choice_types:
            df_grid_with_choices.loc[house_idx, choice_type] = choices_for_grid[house_idx][choice_type]
    rename_map = {
        "monovalent": "HP",
        "heating_rod": "HP+EH",
        "hybrid": "Hyb",
        "household": "-",
        "household+pv": "PV",
        "household+pv+battery": "PV+Bat",
        "household+e_mobility": "EMob",
        "household+pv+e_mobility": "PV+EMob",
        "household+pv+battery+e_mobility": "PV+Bat+EMob",
        "tabula_standard": "no",
        "tabula_retrofit": "ret",
        "tabula_adv_retrofit": "adv-ret",
        "heat_supply_choice": "heating technology",
        "electricity_system_choice": "electrical technology",
        "construction_type_choice": "building retrofit"
    }
    from hps_grid_interaction.plotting import plot_loadflow
    fig, ax = plt.subplots(3, 1,
                           figsize=get_figure_size(n_columns=1, height_factor=0.7 * 3))
    for idx, choice_type in enumerate(choice_types):
        unique_labels = df_grid_with_choices.loc[:, choice_type].unique()
        # Convert strings to ints
        for label_idx, unique_label in enumerate(unique_labels):
            df_grid_with_choices.loc[df_grid_with_choices.loc[:, choice_type] == unique_label, choice_type] = label_idx
        df_heatmap = plot_loadflow.convert_grid_df_to_heatmap_df(
            df_grid=df_grid_with_choices, column=choice_type
        )
        n = len(unique_labels)
        cmap = sns.color_palette(palette=EBCColors.ebc_palette_sort_2, n_colors=n)

        ax[idx] = plot_loadflow.plot_heat_map_on_grid_image(
            ax=ax[idx],
            cmap=cmap,
            df=df_heatmap
        )
        ax[idx].set_ylabel(rename_map[choice_type])
        plot_loadflow.add_discrete_colorbar(
            ax=ax[idx],
            labels=[rename_map.get(label, label) for label in unique_labels]
        )
    fig.savefig(save_path)
    plt.close("all")


if __name__ == '__main__':
    plot_cop_motivation()
