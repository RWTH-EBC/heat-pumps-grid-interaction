import pathlib
import json
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
from pydantic import BaseModel
from cycler import cycler
from ebcpy import TimeSeriesData

from hps_grid_interaction import PLOTS_PATH


class EBCColors:
    dark_red = [172 / 255, 43 / 255, 28 / 255]
    red = [221 / 255, 64 / 255, 45 / 255]
    light_red = [235 / 255, 140 / 255, 129 / 255]
    green = [112 / 255, 173 / 255, 71 / 255]
    light_grey = [217 / 255, 217 / 255, 217 / 255]
    grey = [157 / 255, 158 / 255, 160 / 255]
    dark_grey = [78 / 255, 79 / 255, 80 / 255]
    light_blue = [157 / 255, 195 / 255, 230 / 255]
    blue = [0 / 255, 84 / 255, 159 / 255]
    ebc_palette_sort_1 = [
        dark_red,
        red,
        light_red,
        dark_grey,
        grey,
        light_grey,
        blue,
        light_blue,
        green,
    ]
    ebc_palette_sort_2 = [
        blue,
        red,
        grey,
        green,
        dark_red,
        dark_grey,
        light_red,
        light_blue,
        light_grey,
    ]
    ebc_palette_sort_3 = [
        green,
        light_blue,
        blue,
        grey,
        dark_grey,
        light_red,
        red,
        dark_red,
        light_grey,
    ]


class PlotVariableConfig(BaseModel):
    label: str = None
    unit: str = None
    factor: float = 1
    offset: float = 0


class PlotConfig(BaseModel):
    variables: Dict[str, PlotVariableConfig]
    rcParams: Dict

    def __init__(self, **data):
        super().__init__(**data)
        self.update_rc_params(rcParams=self.rcParams)

    def get_variable(self, variable: str) -> PlotVariableConfig:
        return self.variables.get(variable, PlotVariableConfig(name=variable))

    def update_config(self, config):
        for key, variable in config.get("variables", {}).items():
            self.variables[key] = PlotVariableConfig(**variable)
        self.update_rc_params(rcParams=config.get("rcParams", {}))

    def update_rc_params(self, rcParams: dict):
        if "axes.prop_cycle" not in rcParams:
            rcParams["axes.prop_cycle"] = cycler(color=EBCColors.ebc_palette_sort_2)
        self.rcParams.update(rcParams)
        plt.rcParams.update(self.rcParams)

    def apply_factor_and_offset(self, df):
        df = df.copy()
        if isinstance(df, TimeSeriesData):
            for variable, config in self.variables.items():
                if variable not in df.get_variable_names():
                    continue
                df.loc[:, (variable, "raw")] /= config.factor
                df.loc[:, (variable, "raw")] -= config.offset
        else:
            for variable, config in self.variables.items():
                if variable not in df.columns:
                    continue
                df.loc[:, variable] /= config.factor
                df.loc[:, variable] -= config.offset

        return df

    def get_label_and_unit(self, variable):
        var_obj = self.get_variable(variable)
        if var_obj.label is None:
            return variable
        if var_obj.unit is None:
            return f"{var_obj.label} in [not set]"
        return f"{var_obj.label} in {var_obj.unit}"

    @classmethod
    def parse_json_file(cls, json_file: pathlib.Path):
        with open(json_file, "r") as file:
            return cls.model_validate(json.load(file))

    @classmethod
    def load_default(cls):
        return cls.parse_json_file(PLOTS_PATH.joinpath("hybrid_plot_config.json"))


def load_plot_config(plot_config: dict = None):
    default_plt_config = PlotConfig.load_default()

    if isinstance(plot_config, str):
        with open(plot_config, "r") as file:
            plot_config = json.load(file)
    if isinstance(plot_config, dict):
        default_plt_config.update_config(plot_config)
    return default_plt_config
