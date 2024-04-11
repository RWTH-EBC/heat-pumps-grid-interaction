import datetime
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from hps_grid_interaction.plotting.config import PlotConfig, load_plot_config
import seaborn as sns

from ebcpy import TimeSeriesData


def get_max_int_gain_ratio(df: TimeSeriesData):
    df = df.to_df()
    df.loc[:, "ratio"] = df.loc[:, "outputs.building.QIntGain[1].value"] / df.loc[:, "systemParameters.QBui_flow_nominal[1]"]
    print(df.loc[:, "ratio"].max() * 100)


def plot_variable_correlation(df: pd.DataFrame, variables: List[str], plot_config: PlotConfig):
    def generate_combinations(strings):
        import itertools
        unique_combinations = set()

        for pair in itertools.product(strings, repeat=2):
            if pair[0] != pair[1]:
                sorted_pair = tuple(sorted(pair))
                unique_combinations.add(sorted_pair)

        return list(unique_combinations)

    df = df.loc[:, variables]
    df = plot_config.apply_factor_and_offset(df=df)

    variable_combinations = generate_combinations(variables)
    for variable_combination in variable_combinations:
        var_x, var_y = variable_combination
        plt.figure()
        plt.scatter(df.loc[:, var_x], df.loc[:, var_y])
        plt.ylabel(plot_config.get_label_and_unit(var_y))
        plt.xlabel(plot_config.get_label_and_unit(var_x))
    #plt.show()


if __name__ == '__main__':
    PLOT_CONFIG = load_plot_config("plots/hybrid_plot_config.json")
    from hps_grid_interaction import RESULTS_BES_FOLDER
    DF = TimeSeriesData(RESULTS_BES_FOLDER.joinpath("HybridGEG_oldbuildings", "SimulationResults", "0_TRY2015_523845130645_Jahr_EFH1970_standard_SingleDwelling_DHWCalc_None.mat"))
    DF.loc[:, ("userProfiles.useProBus.absIntGai", "raw")] = DF.loc[:, ("userProfiles.useProBus.absIntGaiRad", "raw")] + DF.loc[:, ("userProfiles.useProBus.absIntGaiConv", "raw")]
    VARIABLES = [
        "outputs.building.QIntGain[1].value",
        "outputs.DHW.Q_flow.value",
        "userProfiles.useProBus.absIntGai"
    ]
    #get_max_int_gain_ratio(df=DF)
    plot_variable_correlation(df=DF, variables=VARIABLES, plot_config=PLOT_CONFIG)
