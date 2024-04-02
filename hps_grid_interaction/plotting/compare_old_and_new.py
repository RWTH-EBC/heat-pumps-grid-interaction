import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from hps_grid_interaction.utils import load_outdoor_air_temperature


def plot_old_and_new(path_new: Path, path_old: Path, save_path: Path):
    df_new = pd.read_excel(path_new.joinpath("Results.xlsx"))
    df_old = pd.read_excel(path_old.joinpath("Results.xlsx"))

    os.makedirs(save_path, exist_ok=True)
    for col in df_new:
        if col.endswith(".value") or col not in df_old.columns:
            continue
        plt.figure()
        plt.scatter(df_old.loc[:, col], df_new.loc[:, col], color="blue")
        plt.xlabel("Old")
        plt.ylabel("New")
        plt.title(col)
        plt.savefig(save_path.joinpath(col.replace(".", "_") + ".png"))
        plt.close("all")


def get_t_m_old_new(path_new: Path, path_old: Path, save_path: Path):
    results_new = get_results_for_single_study(path_new)
    results_old = get_results_for_single_study(path_old)
    T_m_new = [results_new[key]["T_m"] for key in sorted(results_new.keys())]
    P_max_new = [results_new[key]["P_max"] for key in sorted(results_new.keys())]
    P_max_old = [results_old[key]["P_max"] for key in sorted(results_old.keys())]
    T_m_old = [results_old[key]["T_m"] for key in sorted(results_old.keys())]
    plt.figure()
    plt.scatter(T_m_old, T_m_new, color="blue")
    plt.xlabel("Old")
    plt.ylabel("New")
    plt.title("TMean")
    plt.savefig(save_path.joinpath("MeanTemperature.png"))
    plt.figure()
    plt.scatter(P_max_old, P_max_new, color="blue")
    plt.xlabel("Old")
    plt.ylabel("New")
    plt.title("Max")
    plt.savefig(save_path.joinpath("PMax.png"))
    plt.close("all")


def get_results_for_single_study(path: Path):
    results = {}
    from ebcpy import TimeSeriesData

    t_oda = load_outdoor_air_temperature()
    real_winter = t_oda.loc[:, "hydraulic.generation.weaBus.TDryBul"].values < 10

    for file in os.listdir(path.joinpath("SimulationResults")):
        if not file.endswith(".hdf"):
            continue
        path_file = path.joinpath("SimulationResults", file)
        df = TimeSeriesData(path_file).to_df().loc[86400 * 2:]
        T_m = calculate_mean_supply_temperature(df=df, real_winter=real_winter)
        P_max = calculate_max_generation(df=df)
        results[int(path_file.stem.split("_")[0])] = dict(T_m=T_m, P_max=P_max)
    return results


def csv_inputs(path_new: Path, path_old: Path, save_path: Path):
    maxes_new = {}
    maxes_old = {}
    for file in os.listdir(path_new.joinpath("csv_files")):
        if file.endswith(".csv"):
            maxes_new[int(file.split("_")[0])] = pd.read_csv(path_new.joinpath("csv_files", file), index_col=0).loc[:, "household"].max()

    for file in os.listdir(path_old.joinpath("csv_files")):
        if file.endswith(".csv"):
            df_old = pd.read_csv(path_old.joinpath("csv_files", file), index_col=0)
            if "0" in df_old.columns:
                maxes_old[int(file.split("_")[0])] = df_old.loc[:, "0"].max()
            else:
                maxes_old[int(file.split("_")[0])] = df_old.loc[:, "outputs.hydraulic.gen.PEleHeaPum.value"].max()
    plt.figure()
    plt.scatter([maxes_old[k] for k in sorted(maxes_old.keys())], [maxes_new[k] for k in sorted(maxes_new.keys())])
    plt.xlabel("Old")
    plt.ylabel("New")
    plt.title("Max")
    plt.savefig(save_path.joinpath("PMaxCSV.png"))


def plot_worst_day(path_new: Path, path_old: Path, save_path: Path):
    maxes_new = []
    maxes_old = []
    start_day = 28 * 86400
    end_day = 29 * 86400
    for file in os.listdir(path_new.joinpath("csv_files")):
        if file.endswith(".csv") and "_standard_" in file:
            maxes_new.append(pd.read_csv(path_new.joinpath("csv_files", file), index_col=0).loc[start_day:end_day, "household"])

    for file in os.listdir(path_old.joinpath("csv_files")):
        if file.endswith(".csv") and "_standard_" in file:
            df_old = pd.read_csv(path_old.joinpath("csv_files", file), index_col=0)
            if "0" in df_old.columns:
                maxes_old.append(df_old.loc[start_day:end_day, "0"])
            else:
                maxes_old.append(df_old.loc[start_day:end_day, "outputs.hydraulic.gen.PEleHeaPum.value"])
    plt.figure()
    sum_old = np.sum(maxes_old, axis=0)
    sum_new = np.sum(maxes_new, axis=0)
    plt.plot(sum_old, label="old", color="blue")
    plt.plot(sum_new, label="new", color="red")
    plt.xlabel("Time")
    plt.ylabel("P in kW")
    plt.savefig(save_path.joinpath("PMaxCSV_worst_day.png"))


def plot_e_mobility_over_t_ambient(save_path: Path):
    from hps_grid_interaction import E_MOBILITY_DATA
    all_profiles = []
    for file in os.listdir(E_MOBILITY_DATA):
        all_profiles.append(pd.read_csv(E_MOBILITY_DATA.joinpath(file), index_col=0).values)
    t_oda = load_outdoor_air_temperature()
    from ebcpy.preprocessing import convert_index_to_datetime_index
    t_oda_time_index = convert_index_to_datetime_index(t_oda.copy(),
                                                       unit_of_index="h",
                                                       origin=datetime.datetime(2023, 1, 1))

    plt.figure()
    plt.scatter(t_oda.values[1:, 0], np.sum(all_profiles, axis=0))
    plt.xlabel("TOda in °C")
    plt.ylabel("P EMobility in kW")
    plt.savefig(save_path.joinpath("PEMobility.png"))

    plt.figure()
    plt.scatter(t_oda.values[:, 0], t_oda_time_index.index.hour)
    plt.xlabel("TOda in °C")
    plt.ylabel("Time of day")
    plt.savefig(save_path.joinpath("TimeODA.png"))


def calculate_max_generation(df: pd.DataFrame):
    HP = df.loc[:, "outputs.hydraulic.gen.PEleHeaPum.value"].values
    if "outputs.hydraulic.gen.PEleEleHea.value" in df:
        EH = df.loc[:, "outputs.hydraulic.gen.PEleEleHea.value"].values
    elif "outputs.hydraulic.gen.PEleHeaRod.value" in df:
        EH = df.loc[:, "outputs.hydraulic.gen.PEleHeaRod.value"].values
    else:
        EH = np.zeros(len(HP))
    return np.max(EH + HP)


def calculate_mean_supply_temperature(df: pd.DataFrame, real_winter):
    T = df.loc[real_winter, "hydraulic.distribution.sigBusDistr.TStoBufTopMea"].values
    Q = df.loc[real_winter, "outputs.building.QTraGain[1].integral"].values
    return np.sum(T * Q) / Q.sum() - 273.15


if __name__ == '__main__':
    save_path = Path(r"D:\00_temp\plots\monovalent_hr")
    path_new = Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations\MonovalentWeather_altbau_HR")
    path_old = Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\01_BESMod_Simulationen\Monovalent_altbau_HR")
    #plot_old_and_new(path_new=path_new, path_old=path_old, save_path=save_path)
    #get_t_m_old_new(path_new=path_new, path_old=path_old, save_path=save_path)
    #csv_inputs(path_new=path_new, path_old=path_old, save_path=save_path)
    # plot_worst_day(path_new=path_new, path_old=path_old, save_path=save_path)
    plot_e_mobility_over_t_ambient(save_path=save_path.parent)
