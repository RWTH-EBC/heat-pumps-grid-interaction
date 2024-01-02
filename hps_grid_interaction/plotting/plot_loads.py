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
    plt.show()


def plot_days_in_year_over_hours_in_day(df: pd.DataFrame, variables: List[str], plot_config: PlotConfig):
    df = df.loc[:86400*365 - 1]
    df.to_datetime_index(origin=datetime.datetime(2023, 1, 1))
    df = df.to_df()

    df = df.loc[:, variables]
    df = plot_config.apply_factor_and_offset(df=df)

    # Extract day of the year and minute of the day from the datetime index
    df['day_of_year'] = df.index.dayofyear
    df['hour_of_day'] = df.index.hour + df.index.minute / 60 + 1
    for variable in variables:
        # Pivot the DataFrame to create a matrix for the heatmap
        heatmap_data = df.pivot(index='day_of_year', columns='hour_of_day', values=variable)

        # Create the heatmap
        plt.figure()  #figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap='rocket_r', cbar_kws={'label': plot_config.get_label_and_unit(variable)})

        # Customize the plot
        plt.title(plot_config.get_label_and_unit(variable))
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Year')
        plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    PLOT_CONFIG = load_plot_config("plots/hybrid_plot_config.json")

    DF = TimeSeriesData(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations\HybridGEG_altbau\SimulationResults\0_TRY2015_523845130645_Jahr_EFH1970_standard_SingleDwelling_DHWCalc_None.mat")
    DF.loc[:, ("userProfiles.useProBus.absIntGai", "raw")] = DF.loc[:, ("userProfiles.useProBus.absIntGaiRad", "raw")] + DF.loc[:, ("userProfiles.useProBus.absIntGaiConv", "raw")]
    VARIABLES = [
        "outputs.building.QIntGain[1].value",
        "outputs.DHW.Q_flow.value",
        "userProfiles.useProBus.absIntGai"
    ]
    #get_max_int_gain_ratio(df=DF)
    #plot_days_in_year_over_hours_in_day(df=DF, variables=VARIABLES, plot_config=PLOT_CONFIG)
    plot_variable_correlation(df=DF, variables=VARIABLES, plot_config=PLOT_CONFIG)
