"""Module with functions to create the table in the Appendix"""
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from hps_grid_interaction.plotting import get_figure_size
from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction import PLOTS_PATH
from ebcpy import TimeSeriesData


def get_cop(hold_last: bool = False):
    if hold_last:
        data = {
            -20: [2.39, 2.39, 2.05, 1.62, 1.60, float('nan')],
            -15: [2.66, 2.66, 2.27, 1.80, 1.81, 1.60],
            -10: [2.97, 2.97, 2.52, 2.01, 2.00, 1.84],
            -7: [3.16, 3.16, 2.67, 2.13, 2.11, 1.99],
            2: [4.46, 4.46, 3.55, 2.60, 2.48, 2.24],
            7: [5.31, 5.31, 4.14, 2.97, 2.81, 2.53],
            10: [5.85, 5.85, 4.52, 3.24, 3.00, 2.69],
            20: [8.55, 8.55, 6.18, 4.54, 3.74, 3.30],
            30: [8.87, 8.87, 9.07, 6.03, 4.74, 4.09],
            35: [8.87, 8.87, 9.07, 6.03, 4.74, 4.09]
        }
        index = [20, 35, 45, 55, 65, 70]
    else:
        data = {
            -20: [2.39, 2.05, 1.62, 1.60, float('nan')],
            -15: [2.66, 2.27, 1.80, 1.81, 1.60],
            -10: [2.97, 2.52, 2.01, 2.00, 1.84],
            -7: [3.16, 2.67, 2.13, 2.11, 1.99],
            2: [4.46, 3.55, 2.60, 2.48, 2.24],
            7: [5.31, 4.14, 2.97, 2.81, 2.53],
            10: [5.85, 4.52, 3.24, 3.00, 2.69],
            20: [8.55, 6.18, 4.54, 3.74, 3.30],
            # 30: [8.87, 9.07, 6.03, 4.74, 4.09],
            # 35: [8.87, 9.07, 6.03, 4.74, 4.09]
        }
        index = [35, 45, 55, 65, 70]
    df = pd.DataFrame(data, index=index)
    df = df.sort_index(ascending=False)
    return df


def _interpolate_cop(TSupply, TOda, df_cop):
    if TSupply not in df_cop.index:
        df_cop.loc[TSupply] = np.NAN
        df_cop = df_cop.sort_index(ascending=False)
        df_cop = df_cop.interpolate()
    if TOda not in df_cop.columns:
        df_cop.loc[TSupply, TOda] = np.NAN
        df_cop = df_cop.sort_index(axis=1)
        df_cop = df_cop.interpolate(axis=1)
    cop = df_cop.loc[TSupply, TOda]
    if pd.isna(cop):
        print(f"{TSupply=}, {TOda=} leads to COP being NAN")
    return cop, df_cop


def create_table_design_capacities():
    base_path = pathlib.Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\01_BESMod_Simulationen")
    df_cop = get_cop(hold_last=True)

    # Create index levels
    buildings = ['EFH', 'MFH (6 WE)', 'MFH (10 WE)']
    years = [1950, 1960, 1970, 1980, 2010]
    scenarios = ['tabula_standard', 'tabula_retrofit', 'tabula_adv_retrofit']
    design_values = ['P_PV', 'Q_HL', 'P_Mon', 'P_Biv', 'P_Hyb', 'P_EH']
    df_app = pd.DataFrame(
        index=pd.MultiIndex.from_product([buildings, years], names=['Building', 'Year']),
        columns=pd.MultiIndex.from_product([scenarios, design_values], names=['Scenario', 'Design'])
    )

    PV_MPP = "electrical.generation.PEleMaxPowPoi[1]"
    QEleHea_nominal = "hydraulic.generation.parHeaPum.QSec_flow_nominal"
    etaEleHea = "hydraulic.generation.parEleHea.eta"
    QHeaLoa = "hydraulic.generation.parHeaPum.QGen_flow_nominal"
    QHeaLoaAtBiv = "hydraulic.generation.parHeaPum.QPri_flow_nominal"
    THydAtBiv_nominal = "THydAtBiv_nominal"
    TBiv = "parameterStudy.TBiv"

    for case in [
        "HybridWeather_newbuildings",
        "HybridWeather_oldbuildings",
        "MonovalentWeather_newbuildings",
        "MonovalentWeather_oldbuildings",
        "MonovalentWeather_oldbuildings_HR",
        "MonovalentWeather_newbuildings_HR",
    ]:
        path = base_path.joinpath(case)
        path_sim = path.joinpath("SimulationResults")
        df_sim = pd.read_excel(
            path.joinpath("MonteCarloSimulationInput.xlsx"),
            sheet_name="Sheet1"
        )
        # First monovalent+EH
        for file in os.listdir(path_sim):
            if not file.endswith(".mat"):
                continue
            row = df_sim.loc[int(file.split("_")[0])]
            idx_loc = (row["Gebäudetyp"], row["Baujahr"])
            retrofit = row["construction_type"]
            if case.endswith("HR"):
                if not pd.isna(df_app.loc[idx_loc, (retrofit, "P_PV")]):
                    continue
                # Only parameters for design:
                tsd = load_tsd(path_sim.joinpath(file), variable_names=[
                    PV_MPP,
                    QEleHea_nominal,
                    etaEleHea,
                    QHeaLoa,
                    QHeaLoaAtBiv,
                    THydAtBiv_nominal,
                    TBiv,
                ])
                if tsd is None:
                    continue
                COP, df_cop = _interpolate_cop(tsd[THydAtBiv_nominal] - 273.15, tsd[TBiv] - 273.15, df_cop)
                df_app.loc[idx_loc, (retrofit, "P_PV")] = tsd[PV_MPP] / 1000
                df_app.loc[idx_loc, (retrofit, "Q_HL")] = tsd[QHeaLoa] / 1000
                df_app.loc[idx_loc, (retrofit, "P_Biv")] = tsd[QHeaLoaAtBiv] / COP / 1000
                df_app.loc[idx_loc, (retrofit, "P_EH")] = tsd[QEleHea_nominal] / tsd[etaEleHea] / 1000
            elif case.startswith("Hybrid"):
                if not pd.isna(df_app.loc[idx_loc, (retrofit, "P_Hyb")]):
                    continue
                # Only parameters for design:
                tsd = load_tsd(path_sim.joinpath(file), variable_names=[
                    QHeaLoaAtBiv,
                    THydAtBiv_nominal,
                    TBiv,
                ])
                if tsd is None:
                    continue
                COP, df_cop = _interpolate_cop(tsd[THydAtBiv_nominal] - 273.15, tsd[TBiv] - 273.15, df_cop)
                df_app.loc[idx_loc, (retrofit, "P_Hyb")] = tsd[QHeaLoaAtBiv] / COP / 1000
            else:
                if not pd.isna(df_app.loc[idx_loc, (retrofit, "P_Mon")]):
                    continue
                # Only parameters for design:
                tsd = load_tsd(path_sim.joinpath(file), variable_names=[
                    QHeaLoaAtBiv,
                    THydAtBiv_nominal,
                    TBiv,
                ])
                if tsd is None:
                    continue
                COP, df_cop = _interpolate_cop(tsd[THydAtBiv_nominal] - 273.15, tsd[TBiv] - 273.15, df_cop)
                df_app.loc[idx_loc, (retrofit, "P_Mon")] = tsd[QHeaLoaAtBiv] / COP / 1000
    df_app.to_excel(PLOTS_PATH.joinpath("design_capacities.xlsx"))


def load_tsd(path, variable_names):
    # Only parameters for design:
    try:
        return TimeSeriesData(
            path, variable_names=variable_names
        ).to_df().iloc[-1].to_dict()
    except (OSError, FileNotFoundError):
        return None


def plot_cop_map():
    df = get_cop()
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size(n_columns=1.3))
    sns.heatmap(
        df, ax=ax, linewidths=0.5, cmap="rocket_r",
        zorder=1, linecolor='black'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Outdoor air temperature in °C")
    ax.set_ylabel("Supply temperature in °C")
    ax.set_title("$COP$ in -")
    fig.tight_layout()
    fig.savefig(PLOTS_PATH.joinpath("COP_map.png"))
    plt.show()


if __name__ == '__main__':
    PlotConfig.load_default()
    # plot_cop_map()
    create_table_design_capacities()
