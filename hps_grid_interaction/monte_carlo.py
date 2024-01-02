import json
import logging
import os
import pickle
from pathlib import Path
from random import choices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ebcpy import TimeSeriesData

from hps_grid_interaction.plotting.config import PlotConfig
from utils import load_outdoor_air_temperature, get_sarnierungsquoten
from emissions import COLUMNS_EMISSIONS, get_emission_options

COLUMNS_GEG = ["percent_renewables", "QRenewable", "QBoi"]


def load_data(path: Path):
    df = TimeSeriesData(path).to_df()
    df.index.name = "time"
    return df


def _draw_uncertain_choice(construction_type: dict, system_type: dict, possible_rows):
    retrofit_choice = choices(list(construction_type.keys()), list(construction_type.values()), k=1)[0]
    system_choice = choices(list(system_type.keys()), list(system_type.values()), k=1)[0]

    mask = (
            (possible_rows.loc[:, "construction_type"] == retrofit_choice) &
            (possible_rows.loc[:, "system_type"] == system_choice)
    )
    if not np.any(mask):
        raise KeyError("No mask fitted, something went wrong")
    return possible_rows.loc[mask, "simulation_result"].values[0]


def load_function_kwargs_prior_to_monte_carlo(hybrid: Path, monovalent: Path, grid_case: str):
    df_sim_hyb = pd.read_excel(
        hybrid.joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
        sheet_name="Sheet1",
        index_col=0
    )
    df_sim_mon = pd.read_excel(
        monovalent.joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
        sheet_name="Sheet1",
        index_col=0
    )

    def _load_csv_data(path: Path, system_type: str):
        data = {}
        ordered_sim_results = {}
        for file in os.listdir(path.joinpath("csv_files")):
            if file.endswith(".csv"):
                df = pd.read_csv(path.joinpath("csv_files", file)).set_index("Time")
                assert len(df.columns) == 1, "Only one column is expected"
                #file_path = system_type + "_" + file  # Enable if the total path fails to work
                file_path = str(path.joinpath("csv_files", file).absolute())
                data[file_path] = df.loc[:, df.columns[0]].values
                n_sim = int(file.split("_")[0])
                ordered_sim_results[n_sim] = file_path
        ordered_sim_results = {key: ordered_sim_results[key] for key in sorted(ordered_sim_results)}
        return data, ordered_sim_results

    time_series_data, sim_results = _load_csv_data(hybrid,  system_type="hybrid")
    df_sim_hyb.loc[:, "simulation_result"] = list(sim_results.values())
    df_sim_hyb.loc[:, "system_type"] = "hybrid"
    time_series_data_m, sim_results_mon = _load_csv_data(monovalent, system_type="monovalent")
    df_sim_mon.loc[:, "simulation_result"] = list(sim_results_mon.values())
    df_sim_mon.loc[:, "system_type"] = "monovalent"
    df_sim = pd.concat([df_sim_hyb, df_sim_mon])
    time_series_data.update(time_series_data_m)
    del time_series_data_m, sim_results_mon, sim_results, df_sim_mon

    df_grid = pd.read_excel(
        Path(__file__).parent.joinpath("Kerber_Vorstadtnetz.xlsx"),
        sheet_name=f"Kerber Netz {grid_case.capitalize()}",
        index_col=0
    )
    func_kwargs = dict(df_sim=df_sim, df_grid=df_grid, time_series_data=time_series_data)
    return func_kwargs


def run_monte_carlo_sim(function_kwargs: dict, n_monte_carlo=1000):
    hybrid_grid, _, _ = run_single_grid_simulation(hybrid_quota=100, **function_kwargs)  # to populate dict
    max_data = {point: {"Hybrid": [], "Monovalent": []} for point in hybrid_grid}
    sum_data = {point: {"Hybrid": [], "Monovalent": []} for point in hybrid_grid}
    tsd_data = {point: {"Hybrid": [], "Monovalent": []} for point in hybrid_grid}
    monte_carlo_history = {"Hybrid": [], "Monovalent": []}
    for _ in range(n_monte_carlo):
        for tech, quota in zip(["Hybrid", "Monovalent"], [100, 0]):
            grid, _, retrofit_distribution = run_single_grid_simulation(hybrid_quota=quota, **function_kwargs)
            monte_carlo_history[tech].append(retrofit_distribution)
            for point, data in grid.items():
                max_data[point][tech].append(data.max())
                sum_data[point][tech].append(data.sum() * 0.25)
                tsd_data[point][tech].append(data)

    monte_carlo_data = {
        "max": max_data,
        "sum": sum_data,
        "tsd_data": tsd_data,
        "simulations": monte_carlo_history
    }

    return monte_carlo_data


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


def plot_monte_carlo_violin(data: dict, metric: str, save_path: Path):
    y_label, factor = get_label_and_factor(metric)
    max_data = data[metric]
    # Violins
    plt.figure()
    for point, values in max_data.items():
        fig, ax = plt.subplots(1, 2)
        for _ax, label in zip(ax, ["Hybrid", "Monovalent"]):
            data = values[label]
            violin_settings = dict(points=100, showextrema=True)
            _ax.violinplot(np.array(data) * factor, **violin_settings)
            _ax.set_xticks(np.arange(1, len([label]) + 1), labels=[label])
            _ax.set_ylabel(y_label)
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


def plot_monte_carlo_bars(data: dict, metric: str, save_path: Path):
    y_label, factor = get_label_and_factor(metric)
    max_data = data[metric]
    # Violins
    plt.figure()
    mean_hyb = []
    mean_mon = []
    std_hyb = []
    std_mon = []
    for point, data in max_data.items():
        mean_hyb.append(np.mean(data["Hybrid"]) * factor)
        std_hyb.append(np.std(data["Hybrid"]) * factor)
        mean_mon.append(np.mean(data["Monovalent"]) * factor)
        std_mon.append(np.std(data["Monovalent"]) * factor)
        print(point, (1 - mean_hyb[-1] / mean_mon[-1]) * 100)
    fig, ax = plt.subplots()
    points = list(max_data.keys())
    x_pos = np.arange(len(points))
    bar_args = dict(align='center', ecolor='black', width=0.4)
    ax.bar(x_pos - 0.2, mean_hyb, yerr=std_hyb, color="blue", **bar_args, label="Hybrid")
    ax.bar(x_pos + 0.2, mean_mon, yerr=std_mon, color="red", **bar_args, label="Monovalent")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.legend()
    ax.set_xticklabels(points)
    ax.yaxis.grid(True)

    fig.tight_layout()
    fig.savefig(save_path.joinpath(f"monte_carlo_{metric}.png"))
    plt.close("all")


def plot_cop_motivation():
    plt.rcParams.update({"figure.figsize": [6.24 * 1.2, 5.78 / 1.3]})
    tsd35 = TimeSeriesData(r"D:\04_git\design-rule-synthesizer\studies\hybrid_hps\cop_data\GetCOPCurve308.mat").to_df()
    tsd70 = TimeSeriesData(r"D:\04_git\design-rule-synthesizer\studies\hybrid_hps\cop_data\GetCOPCurve343.mat").to_df()
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


def run_single_grid_simulation(
        df_sim: pd.DataFrame, df_grid: pd.DataFrame,
        hybrid_quota: int, time_series_data: dict, sarnierungsquoten: dict
):
    grid = {key: 0 for key in [f"AP_{i + 1}" for i in range(10)]}
    csv_files_for_acs = []
    _resulting_retrofit_distribution = []
    for house_index, row in df_grid.iterrows():
        building_type = row["Gebäudetyp"].split(" ")[0]  # In case of MFH (..)
        sim_result_name = _draw_uncertain_choice(
            construction_type=sarnierungsquoten[building_type][row["Baujahr"]],
            system_type={"monovalent": 100 - hybrid_quota, "hybrid": hybrid_quota},
            possible_rows=df_sim.loc[house_index]
        )
        _resulting_retrofit_distribution.append(sim_result_name)
        grid[f"AP_{row['Anschlusspunkt'].split('-')[0]}"] += time_series_data[sim_result_name]
        csv_name = "csv_files//" + "_".join(sim_result_name.split("_")[1:])
        csv_files_for_acs.append(csv_name)
    grid["ONT"] = np.sum(list(grid.values()), axis=0)  # Build overall sum at ONT
    df_csv = df_grid.copy()
    df_csv.loc[:, "Ergebnisse_EBC"] = csv_files_for_acs
    return grid, df_csv, _resulting_retrofit_distribution


def argmean(arr):
    # Calculate the mean of the array
    mean = np.mean(arr)
    # Calculate the absolute differences between each element and the mean
    abs_diff = np.abs(arr - mean)
    # Find the index of the element with the smallest absolute difference
    return np.argmin(abs_diff)


def get_short_case_name(tech: str, case_name: str):
    return tech[:3] + "_" + "_".join(s[:5] for s in case_name.split("_"))


def plot_and_export_single_monte_carlo(
        data, metric: str, save_path: Path,
        case_name: str, grid_case: str, df_sim: pd.DataFrame
):
    arg_function = argmean
    export_data = {}
    emissions_data = {}
    for tech in ["Hybrid", "Monovalent"]:
        arg = arg_function(data[metric]["ONT"][tech])
        # Save in excel for Lastflusssimulation:
        to_grid_simulation = data["simulations"][tech][arg]

        df_lastfluss = pd.read_excel(
            Path(__file__).parent.joinpath("Kerber_Vorstadtnetz.xlsx"),
            sheet_name=f"{grid_case}_lastfluss_template",
            index_col=0
        )
        df_lastfluss["Wärmepumpenstrom-Zeitreihe"] = to_grid_simulation
        workbook_name = save_path.parent.joinpath("LastflussSimulationen.xlsx")
        short_sheet_name = get_short_case_name(tech=tech, case_name=case_name)
        save_excel(df=df_lastfluss, path=workbook_name, sheet_name=short_sheet_name)
        export_data[tech] = {
            "max": {point: data["max"][point][tech][arg] for point in data["max"].keys()},
            "sum": {point: data["sum"][point][tech][arg] for point in data["sum"].keys()}
        }
        save_excel(df=pd.DataFrame(export_data[tech]), path=save_path.parent.joinpath(f"MonteCarloResults.xlsx"),
                   sheet_name=f"{metric}_{short_sheet_name}")
        plt.figure()
        plt.suptitle(tech)
        for point in data["tsd_data"].keys():
            tsd = data["tsd_data"][point][tech][arg]
            plt.plot(tsd, label=point)
            plt.ylabel(get_label_and_factor("value")[0])
        plt.legend(loc="upper right", ncol=2)
        plt.savefig(save_path.joinpath(f"time_series_plot_{tech}_{metric}.png"))
        sim_results = data["simulations"][tech][arg]
        tech_mask = df_sim.loc[:, "system_type"] == tech.lower()
        df_tech = df_sim.loc[tech_mask]
        columns = COLUMNS_EMISSIONS + COLUMNS_GEG
        sum_cols = {col: 0 for col in columns}
        for sim_result in sim_results:
            row = df_tech.loc[df_tech.loc[:, "simulation_result"] == sim_result]
            for col in columns:
                sum_cols[col] += row[col].values[0]
        emissions_data[tech] = sum_cols
    return {"grid": export_data, "emissions": emissions_data}


def save_excel(df, path, sheet_name):
    if path.exists():
        with pd.ExcelWriter(path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        df.to_excel(path, sheet_name=sheet_name)


def run_save_and_plot_monte_carlo(
        grid_case: str, with_hr: bool, res: dict,
        quota_name: str, load: bool = False,
        n_monte_carlo: int = 1000, extra_case_name_hybrid: str = ""
):
    all_results = res
    sarnierungsquoten = get_sarnierungsquoten(assumption=quota_name)

    with_hr_str = "_HR" if with_hr else ""
    case_name = f"{grid_case}{with_hr_str}_{quota_name}"

    simulation_results_path = Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations")
    save_path = simulation_results_path.joinpath(f"MonteCarloResults_{case_name}")
    pickle_path = save_path.joinpath(f"monte_carlo_{case_name}.pickle")

    hybrid_path = simulation_results_path.joinpath(f"Hybrid{extra_case_name_hybrid}_{grid_case}")
    if not hybrid_path.exists():
        print(f"Did not find {hybrid_path}. Skipping case {case_name}")
        return res
    kwargs = load_function_kwargs_prior_to_monte_carlo(
        hybrid=hybrid_path,
        monovalent=simulation_results_path.joinpath(f"Monovalent_{grid_case}{with_hr_str}"),
        grid_case=grid_case
    )
    kwargs["sarnierungsquoten"] = sarnierungsquoten

    if load and pickle_path.exists():
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
    else:
        import time
        t0 = time.time()
        data = run_monte_carlo_sim(function_kwargs=kwargs, n_monte_carlo=n_monte_carlo)
        print(f"Simulations took {time.time() - t0} s")

    os.makedirs(save_path, exist_ok=True)
    plot_monte_carlo_bars(data=data, metric="max", save_path=save_path)
    plot_monte_carlo_bars(data=data, metric="sum", save_path=save_path)
    plot_monte_carlo_violin(data=data, metric="max", save_path=save_path)
    plot_monte_carlo_violin(data=data, metric="sum", save_path=save_path)
    export_data = plot_and_export_single_monte_carlo(
        data=data, metric="max", df_sim=kwargs["df_sim"],
        save_path=save_path, case_name=case_name, grid_case=grid_case
    )
    all_results[case_name] = export_data
    with open(pickle_path, "wb") as file:
        pickle.dump(data, file)
    return all_results


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


def run_all_cases(load: bool, extra_case_name_hybrid: str = ""):
    res = {}
    for grid_case in ["altbau", "neubau"]:
        for quota in [
            "average", "no_retrofit",
            #"all_retrofit", "all_adv_retrofit"
        ]:
            if grid_case == "altbau":
                hr = [True, False]
            else:
                hr = [False]
            for with_hr in hr:
                res = run_save_and_plot_monte_carlo(
                    grid_case=grid_case, quota_name=quota,
                    with_hr=with_hr, res=res, load=load,
                    n_monte_carlo=10 if quota != "average" else 1000,
                    extra_case_name_hybrid=extra_case_name_hybrid
                )
    all_results_path = Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations").joinpath("results_to_plot.json")
    with open(all_results_path, "w") as file:
        json.dump(res, file)
    return all_results_path


if __name__ == '__main__':
    PlotConfig.load_default()  # Trigger rc_params
    PATH = run_all_cases(load=True, extra_case_name_hybrid="GEGBiv")
    #plot_results_all_cases(path=Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations\results_to_plot.json"))
