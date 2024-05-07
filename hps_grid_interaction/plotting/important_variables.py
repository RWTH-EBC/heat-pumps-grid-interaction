"""
Plots for simulation analysis
"""
import os
import logging
from pathlib import Path
from typing import List, Union, Dict

import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData

from hps_grid_interaction.plotting.config import load_plot_config


logger = logging.getLogger(__name__)


def get_names_of_plot_variables(
        x_variable: str,
        y_variables: Dict[str, Union[List, str]],
        x_vertical_lines: List[str] = None,
        **kwargs
):
    """
    Small helper function to get all variable names
    to be extracted from the simulation result.
    """
    all_variables = ["weaDat.weaBus.TDryBul"]
    if x_variable != "time":
        all_variables.append(x_variable)
    if x_vertical_lines is not None:
        all_variables.extend(x_vertical_lines)
    for vars in y_variables.values():
        if isinstance(vars, str):
            all_variables.append(vars)
        else:
            all_variables.extend(vars)
    return list(set(all_variables))


def plot_important_variables(
        tsd: TimeSeriesData,
        x_variable: str,
        y_variables: Dict[str, Union[List, str]],
        init_period: float,
        plot_config: dict = None,
        save_path: Path = None,
        show=False,
        scatter: bool = False,
        x_vertical_lines: List[str] = None,
):
    """
    Plot the most important variables during simulation.
    """
    locator = lambda x: (x, "raw") if isinstance(tsd, TimeSeriesData) else lambda x: x

    if x_vertical_lines is None or not scatter:
        x_vertical_lines = []
    if len(x_vertical_lines) > 2:
        raise ValueError("Only two x_vertical_lines are supported")
    plot_config = load_plot_config(plot_config=plot_config)

    tsd = plot_config.apply_factor_and_offset(tsd)
    fig, axes = plt.subplots(nrows=len(y_variables), ncols=1, sharex=True, squeeze=True,
                             figsize=(5.90551 * 1.5, 8.27 * 1.8 * len(y_variables) / 10))
    axes[-1].set_xlabel(plot_config.get_label_and_unit(x_variable))

    for x_vertical_line in x_vertical_lines:
        if tsd.loc[:, locator(x_vertical_line)].std() != 0:
            raise ValueError(f"Given x_vertical_lines {x_vertical_line} in not constant.")

    if x_variable == "time":
        x_values = tsd.loc[init_period:].index
    else:
        x_values = tsd.loc[init_period:, locator(x_variable)]

    for _y_variable, _ax in zip(y_variables, axes):
        _ax.set_ylabel(_y_variable)
        for idx, x_vertical_line in enumerate(x_vertical_lines):
            _ax.axvline(
                tsd.loc[:, locator(x_vertical_line)].mean(),
                label=plot_config.get_variable(x_vertical_line).label,
                color="black"
            )
        plot_variables = y_variables[_y_variable]
        if isinstance(plot_variables, str):
            plot_variables = [plot_variables]
        for plot_variable in plot_variables:
            if scatter:
                _ax.scatter(x_values, tsd.loc[init_period:, locator(plot_variable)],
                            label=plot_config.get_variable(plot_variable).label,
                            s=1)
            else:
                _ax.plot(x_values, tsd.loc[init_period:, locator(plot_variable)],
                         label=plot_config.get_variable(plot_variable).label)
        _ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

    fig.tight_layout()
    fig.align_ylabels()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path)
    if show:
        pass
        #plt.show()
    plt.close("all")


def plot_result(tsd, init_period, result_name, save_path, plot_settings):
    plot_important_variables(
        save_path=save_path.joinpath("plots_time", result_name + ".png"),
        x_variable="time",
        scatter=False,
        tsd=tsd,
        init_period=init_period,
        **plot_settings
    )
    plot_important_variables(
        tsd=tsd,
        save_path=save_path.joinpath("plots_scatter", result_name + ".png"),
        x_variable="weaDat.weaBus.TDryBul",
        scatter=True,
        init_period=init_period,
        **plot_settings
    )
