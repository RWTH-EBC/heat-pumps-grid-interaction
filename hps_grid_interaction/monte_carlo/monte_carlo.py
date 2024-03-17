import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List
from random import choices
import time

import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import get_construction_type_quotas
from hps_grid_interaction.emissions import COLUMNS_EMISSIONS
from hps_grid_interaction.monte_carlo import plots
from hps_grid_interaction import RESULTS_BES_FOLDER, KERBER_NETZ_XLSX, RESULTS_MONTE_CARLO_FOLDER, E_MOBILITY_DATA
from hps_grid_interaction.bes_simulation.simulation import W_to_Wh
logger = logging.getLogger(__name__)


COLUMNS_GEG = ["percent_renewables", "QRenewable", "QBoi"]


class Quotas:

    def __init__(
            self,
            construction_type_quota: str,
            heat_pump_quota: int,
            heating_rod_quota: int,
            hybrid_quota: int,
            pv_quota: int,
            pv_battery_quota: int,
            e_mobility_quota: int,
            n_monte_carlo: int = 1000
    ):
        self.construction_type_quotas = get_construction_type_quotas(assumption=construction_type_quota)
        monoenergetic_quota = (100 - hybrid_quota) * heat_pump_quota / 100
        self.heat_supply_quotas = {
            "monovalent": (100 - heating_rod_quota) * monoenergetic_quota / 100,
            "hybrid": hybrid_quota * heat_pump_quota / 100,
            "gas": (100 - heat_pump_quota),
            "heating_rod": heating_rod_quota * monoenergetic_quota / 100
        }
        self.electricity_system_quotas = {
            "household": (100 - pv_quota - pv_battery_quota) * (1 - e_mobility_quota / 100),
            "household+pv": pv_quota * (1 - e_mobility_quota / 100),
            "household+pv+battery": pv_battery_quota * (1 - e_mobility_quota / 100),
            "household+e_mobility": (100 - pv_quota - pv_battery_quota) * e_mobility_quota / 100,
            "household+pv+e_mobility": pv_quota * e_mobility_quota / 100,
            "household+pv+battery+e_mobility": pv_battery_quota * e_mobility_quota / 100
        }
        self.n_monte_carlo = n_monte_carlo


def _draw_uncertain_choice(quotas: Quotas, building_type: str, year_of_construction: int):

    def _draw_from_dict(d: dict):
        return choices(list(d.keys()), list(d.values()), k=1)[0]

    construction_type_choice = _draw_from_dict(quotas.construction_type_quotas[building_type][year_of_construction])
    heat_supply_choice = _draw_from_dict(quotas.heat_supply_quotas)
    electricity_system_choice = _draw_from_dict(quotas.electricity_system_quotas)

    return {
        "heat_supply_choice": heat_supply_choice,
        "electricity_system_choice": electricity_system_choice,
        "construction_type_choice": construction_type_choice
    }


def _get_time_series_data_for_choices(heat_supplies: dict, house_index: int, all_choices: dict):
    df_sim, time_series_data = heat_supplies[all_choices["heat_supply_choice"]]
    possible_rows = df_sim.loc[house_index]
    if isinstance(possible_rows, pd.Series):
        if not possible_rows["Baujahr"] == 2010:
            raise KeyError("Only one type even though not 2010, something went wrong")
        sim_result_name = possible_rows["simulation_result"]
    else:
        mask = possible_rows.loc[:, "construction_type"] == all_choices["construction_type_choice"]
        if not np.any(mask):
            raise KeyError("No mask fitted, something went wrong")
        sim_result_name = possible_rows.loc[mask, "simulation_result"].values[0]

    return time_series_data[sim_result_name].loc[:, all_choices["electricity_system_choice"]]


def load_function_kwargs_prior_to_monte_carlo(
        hybrid: Path,
        monovalent: Path,
        heating_rod: Path,
        grid_case: str):
    t0 = time.time()

    df_grid = pd.read_excel(
        KERBER_NETZ_XLSX,
        sheet_name=f"Kerber Netz {grid_case.capitalize()}",
        index_col=0
    )
    dfs_e_mobility = {}
    for house_idx in df_grid.index:
        e_mobility_path = E_MOBILITY_DATA.joinpath(f"ev_{house_idx}.csv")
        if not os.path.exists(e_mobility_path):
            e_mobility_path = E_MOBILITY_DATA.joinpath(f"no_ev.csv")
        dfs_e_mobility[house_idx] = pd.read_csv(e_mobility_path, sep=",").loc[:, "Ladestrom [kW]"].values

    def _load_csv_data(_path: Path):
        data = {}
        ordered_sim_results = {}
        for file in os.listdir(_path.joinpath("csv_files")):
            if file.endswith(".csv"):
                file_path = str(_path.joinpath("csv_files", file).absolute())
                df = pd.read_csv(file_path).set_index("Time")
                data[file] = df
                n_sim = int(file.split("_")[0])
                ordered_sim_results[n_sim] = file

        ordered_sim_results = {key: ordered_sim_results[key] for key in sorted(ordered_sim_results)}
        return data, list(ordered_sim_results.values())
    cases = {
        "hybrid": hybrid,
        "monovalent": monovalent,
        "heating_rod": heating_rod
    }
    heat_supplies = {}
    for heat_supply, path_study in cases.items():
        # Should not matter which xlsx is loaded, with or without e-mobility
        df_sim = pd.read_excel(
            path_study.joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
            sheet_name="Sheet1",
            index_col=0
        )
        time_series_data, simulation_result = _load_csv_data(path_study)
        df_sim.loc[:, "simulation_result"] = simulation_result
        for key, df in time_series_data.items():
            house_index = df_sim.loc[df_sim.loc[:, "simulation_result"] == key].index.values[0]
            df.drop(df.head(1).index, inplace=True)
            df.drop(df.tail(1).index, inplace=True)
            for col in df.columns:
                # head and tail are dropped for sim results, head for e_mobility
                df.loc[:, col + "+e_mobility"] = df.loc[:, col] + dfs_e_mobility[house_index][1:]

        heat_supplies[heat_supply] = (df_sim, time_series_data)

    df_sim, time_series_data = heat_supplies["hybrid"]
    gas_time_series_data = {}
    for key, df in time_series_data.items():
        df_gas = df.copy()
        for col in df_gas.columns:
            if col != "heat_supply":
                df_gas.loc[:, col] -= df_gas.loc[:, "heat_supply"]
        gas_time_series_data[key] = df_gas
    heat_supplies["gas"] = (df_sim.copy(), gas_time_series_data)

    func_kwargs = dict(heat_supplies=heat_supplies, df_grid=df_grid)
    logger.info("Loading function inputs took %s s", time.time() - t0)
    return func_kwargs


def run_monte_carlo_sim(quota_cases: Dict[str, Quotas], function_kwargs: dict):
    t0 = time.time()
    max_data = {}
    sum_data = {}
    tsd_data = {}
    choices_for_grid = {quota_name: [] for quota_name in quota_cases.keys()}
    i = 0

    def _get_electricity_sum_in_kWh(data):
        #return data.sum() * W_to_Wh  # Also grid feed in
        return data[data > 0].sum() * W_to_Wh  # Also grid feed in

    for quota_name, quotas in quota_cases.items():
        for _ in range(quotas.n_monte_carlo):
            grid, all_choices = run_single_grid_simulation(quotas=quotas, **function_kwargs)
            choices_for_grid[quota_name].append(all_choices)
            for point, data in grid.items():
                if point not in max_data:
                    max_data[point] = {quota_name: [data.max()]}
                    sum_data[point] = {quota_name: [_get_electricity_sum_in_kWh(data)]}
                    #tsd_data[point] = {quota_name: [data]}
                elif quota_name not in max_data[point]:
                    max_data[point][quota_name] = [data.max()]
                    sum_data[point][quota_name] = [_get_electricity_sum_in_kWh(data)]
                    #tsd_data[point][quota_name] = [data]
                else:
                    max_data[point][quota_name].append(data.max())
                    sum_data[point][quota_name].append(_get_electricity_sum_in_kWh(data))
                    #tsd_data[point][quota_name].append(data)
        i += 1
        logger.info("Ran quota case %s of %s", i, len(quota_cases))

    monte_carlo_data = {
        "max": max_data,
        "sum": sum_data,
        #"tsd_data": tsd_data,
        "choices_for_grid": choices_for_grid
    }
    logger.info("Monte-Carlo simulations took %s s", time.time() - t0)

    return monte_carlo_data


def run_single_grid_simulation(
        heat_supplies: dict,
        df_grid: pd.DataFrame,
        quotas: Quotas,
):
    grid = {key: 0 for key in [f"AP_{i + 1}" for i in range(10)]}

    _resulting_choice_distribution = []
    for house_index, row in df_grid.iterrows():
        building_type = row["Gebäudetyp"].split(" ")[0]  # In case of MFH (..)
        all_choices = _draw_uncertain_choice(
            quotas=quotas,
            building_type=building_type,
            year_of_construction=row["Baujahr"]
        )
        time_series_result = _get_time_series_data_for_choices(
            heat_supplies=heat_supplies,
            house_index=house_index,
            all_choices=all_choices
        )
        _resulting_choice_distribution.append(all_choices)
        grid[f"AP_{row['Anschlusspunkt'].split('-')[0]}"] += time_series_result

    grid["ONT"] = np.sum(list(grid.values()), axis=0)  # Build overall sum at ONT
    return grid, _resulting_choice_distribution


def get_grid_simulation_input_for_choices(
        heat_supplies: dict,
        df_grid: pd.DataFrame,
        choices_for_grid: dict
):
    grid_time_series_data = []
    for house_index, row in df_grid.iterrows():
        time_series_data = _get_time_series_data_for_choices(
            heat_supplies=heat_supplies,
            house_index=house_index,
            all_choices=choices_for_grid[house_index]
        )
        grid_time_series_data.append(time_series_data)
    return grid_time_series_data


def save_grid_time_series_data_to_csv_folder(
        grid_time_series_data: list,
        save_path: Path
):
    os.makedirs(save_path, exist_ok=True)
    _resulting_grid_csv_files = []
    for idx, time_series_data in enumerate(grid_time_series_data):
        csv_file_name = save_path.joinpath(f"{idx}.csv")
        time_series_data.to_csv(csv_file_name, sep=",")
        _resulting_grid_csv_files.append(f".\{save_path.name}\{csv_file_name.name}")
    return _resulting_grid_csv_files


def argmean(arr):
    # Calculate the mean of the array
    mean = np.mean(arr)
    # Calculate the absolute differences between each element and the mean
    abs_diff = np.abs(arr - mean)
    # Find the index of the element with the smallest absolute difference
    return np.argmin(abs_diff)


def get_grid_simulation_case_name(quota_case: str, grid_case: str):
    return quota_case + "_" + grid_case


def plot_and_export_single_monte_carlo(
        quota_cases: Dict[str, Quotas],
        data, metric: str, save_path: Path,
        grid_case: str,
        plots_only: bool,
        heat_supplies: dict, df_grid: pd.DataFrame
):
    arg_function = argmean
    export_data = {}
    emissions_data = {}
    quota_case_grid_simulation_inputs = {}
    quota_case_grid_data = {}
    for quota_case in quota_cases:
        arg = arg_function(data[metric]["ONT"][quota_case])
        # Save in excel for Lastflusssimulation:
        choices_for_grid = data["choices_for_grid"][quota_case][arg]
        grid_time_series_data = get_grid_simulation_input_for_choices(
            heat_supplies=heat_supplies,
            df_grid=df_grid,
            choices_for_grid=choices_for_grid,
        )
        quota_case_grid_data[quota_case] = grid_time_series_data
        if not plots_only:
            csv_file_paths = save_grid_time_series_data_to_csv_folder(
                save_path=save_path.joinpath(f"grid_simulation_{quota_case}"),
                grid_time_series_data=grid_time_series_data
            )
            df_lastfluss = pd.read_excel(
                KERBER_NETZ_XLSX,
                sheet_name=f"{grid_case}_lastfluss_template",
                index_col=0
            )
            df_lastfluss["electricity_time_series_data"] = csv_file_paths
            grid_simulation_case_name = get_grid_simulation_case_name(quota_case=quota_case, grid_case=grid_case)
            workbook_name = save_path.joinpath(f"{grid_simulation_case_name}.xlsx")
            save_excel(df=df_lastfluss, path=workbook_name, sheet_name="lastfluss")
            quota_case_grid_simulation_inputs[quota_case] = str(workbook_name)
        export_data[quota_case] = {
            "max": {point: data["max"][point][quota_case][arg] for point in data["max"].keys()},
            "sum": {point: data["sum"][point][quota_case][arg] for point in data["sum"].keys()}
        }

        # TODO: Fix simulation results for cases
        #quota_case_mask = df_sim.loc[:, "system_type"] == tech.lower()
        #df_quota_case = df_sim.loc[quota_case_mask]
        #columns = COLUMNS_EMISSIONS + COLUMNS_GEG
        #sum_cols = {col: 0 for col in columns}
        #for sim_result in grid_time_series_data:
        #    row = df_quota_case.loc[df_quota_case.loc[:, "simulation_result"] == sim_result]
        #    for col in columns:
        #        sum_cols[col] += row[col].values[0]
        #emissions_data[quota_case] = sum_cols
    #return {"grid": export_data, "emissions": emissions_data}
    plots.plot_time_series(
        quota_case_grid_data=quota_case_grid_data,
        save_path=save_path
    )

    return {"grid": export_data, "grid_simulation_inputs": quota_case_grid_simulation_inputs}


def save_excel(df, path, sheet_name):
    if path.exists():
        with pd.ExcelWriter(path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        df.to_excel(path, sheet_name=sheet_name)

def run_save_and_plot_monte_carlo(
        quota_cases: Dict[str, Quotas],
        grid_case: str,
        save_path: Path,
        load: bool = False,
        extra_case_name: str = "",
):
    os.makedirs(save_path, exist_ok=True)

    kwargs = load_function_kwargs_prior_to_monte_carlo(
        hybrid=RESULTS_BES_FOLDER.joinpath(f"Hybrid{extra_case_name}_{grid_case}"),
        monovalent=RESULTS_BES_FOLDER.joinpath(f"Monovalent{extra_case_name}_{grid_case}"),
        heating_rod=RESULTS_BES_FOLDER.joinpath(f"Monovalent{extra_case_name}_{grid_case}_HR"),
        grid_case=grid_case
    )
    pickle_path = save_path.joinpath(f"monte_carlo_{grid_case}.pickle")
    if load:
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
        logger.info("Loaded pickle data from %s", pickle_path)
    else:
        data = run_monte_carlo_sim(quota_cases=quota_cases, function_kwargs=kwargs)
        with open(pickle_path, "wb") as file:
            pickle.dump(data, file)

    plots.plot_monte_carlo_bars(data=data, metric="max", save_path=save_path, quota_cases=quota_cases)
    plots.plot_monte_carlo_bars(data=data, metric="sum", save_path=save_path, quota_cases=quota_cases)
    plots.plot_monte_carlo_violin(data=data, metric="max", save_path=save_path, quota_cases=quota_cases)
    plots.plot_monte_carlo_violin(data=data, metric="sum", save_path=save_path, quota_cases=quota_cases)
    export_data = plot_and_export_single_monte_carlo(
        quota_cases=quota_cases,
        data=data, metric="max", plots_only=False,
        save_path=save_path, grid_case=grid_case,
        **kwargs
    )

    return export_data


def run_all_cases(load: bool, quota_study: str, extra_case_name_hybrid: str = ""):
    quota_cases = {}
    if quota_study == "av_pv":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=quota, pv_battery_quota=0,
                e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
            )
    elif quota_study == "av_pv_bat":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=quota,
                e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
            )
    elif quota_study == "av_hyb":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
                e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=quota, heating_rod_quota=0
            )
    elif quota_study == "av_heat_pump":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
                e_mobility_quota=0, heat_pump_quota=quota, hybrid_quota=0, heating_rod_quota=0
            )
    elif quota_study == "av_hyb_with_pv_bat":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=100,
                e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=quota, heating_rod_quota=0
            )
    elif quota_study == "av_e_mob_with_pv_bat":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=100,
                e_mobility_quota=quota, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
            )
    elif quota_study == "av_heating_rod":
        for quota in [0, 20, 40, 60, 80, 100]:
            quota_cases[f"{quota_study}_{quota}"] = Quotas(
                construction_type_quota="average", pv_quota=0, pv_battery_quota=100,
                e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=quota
            )
    elif quota_study == "show_extremas":
        quota_cases[f"{quota_study}_" + "GasAverage"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=0, hybrid_quota=0, heating_rod_quota=0
        )
        quota_cases[f"{quota_study}_" + "HybAverage"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=100, heating_rod_quota=0
        )
        quota_cases[f"{quota_study}_" + "MonAverage"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
        )
        quota_cases[f"{quota_study}_" + "HeaRodAverage"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
        quota_cases[f"{quota_study}_" + "HeaRodAverageEMob"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
        quota_cases[f"{quota_study}_" + "HeaRodAverageEMobPV"] = Quotas(
            construction_type_quota="average", pv_quota=100, pv_battery_quota=0,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
        quota_cases[f"{quota_study}_" + "HeaRodAverageEMobPVBat"] = Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
        quota_cases[f"{quota_study}_" + "HeaRodAllRetroEMobPVBat"] = Quotas(
            construction_type_quota="all_retrofit", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
        quota_cases[f"{quota_study}_" + "HeaRodAllAdvRetroEMobPVBat"] = Quotas(
            construction_type_quota="all_adv_retrofit", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )
    #"all_retrofit": Quotas(construction_type_quota="all_retrofit"),
    #"all_adv_retrofit": Quotas(construction_type_quota="all_adv_retrofit"),
    #"no_retrofit": Quotas(construction_type_quota="no_retrofit")
    grid_cases = [
        #"altbau",
        "neubau"
    ]
    all_results = {}
    for grid_case in grid_cases:
        save_path = RESULTS_MONTE_CARLO_FOLDER.joinpath(f"{grid_case.capitalize()}_{quota_study}")
        res = run_save_and_plot_monte_carlo(
            quota_cases=quota_cases,
            grid_case=grid_case,
            save_path=save_path, load=load,
            extra_case_name=extra_case_name_hybrid
        )
        all_results[grid_case] = res
    all_results_path = save_path.joinpath("results_to_plot.json")
    with open(all_results_path, "w") as file:
        json.dump(all_results, file)
    return all_results_path


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    PlotConfig.load_default()  # Trigger rc_params
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_hyb_with_pv_bat")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="show_extremas")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_e_mob_with_pv_bat")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_heat_pump")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_heating_rod")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_pv_bat")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_hyb")
    run_all_cases(load=False, extra_case_name_hybrid="Weather", quota_study="av_pv")
