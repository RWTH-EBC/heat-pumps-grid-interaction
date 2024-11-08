import dataclasses
import itertools
import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Union
from random import choices
import time
import multiprocessing
import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import get_construction_type_quotas
from hps_grid_interaction.monte_carlo import plots
from hps_grid_interaction import (
    RESULTS_BES_FOLDER, KERBER_NETZ_XLSX,
    RESULTS_MONTE_CARLO_FOLDER, E_MOBILITY_DATA,
    DATA_PATH
)
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
            n_monte_carlo: int = 5000
    ):
        self.construction_type_quotas = get_construction_type_quotas(assumption=construction_type_quota)
        self.construction_type_quota = construction_type_quota
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
        assert sum(self.electricity_system_quotas.values()) == 100, "Quotas do not equal 100"
        assert sum(self.heat_supply_quotas.values()) == 100, "Quotas do not equal 100"
        assert all(0 <= v <= 100 for v in self.electricity_system_quotas.values()), \
            f"Quotas are not between 0 and 100: {self.electricity_system_quotas}"
        assert all(0 <= v <= 100 for v in self.heat_supply_quotas.values()), \
            f"Quotas are not between 0 and 100: {self.heat_supply_quotas}"

        for years_dict in self.construction_type_quotas.values():
            for year_dict in years_dict.values():
                assert np.isclose(sum(year_dict.values()), 100, atol=1e-2), \
                    f"Construction type quota '{construction_type_quota}' " \
                    f"does not equal 100: {year_dict}"

        if self._monte_carlo_necessary():
            self.n_monte_carlo = n_monte_carlo
        else:
            self.n_monte_carlo = 1

    def _monte_carlo_necessary(self):
        def _zero_or_hundred(v):
            return v in [0, 100]

        values_ct = []
        for years_dict in self.construction_type_quotas.values():
            for year_dict in years_dict.values():
                values_ct.extend(list(year_dict.values()))
        all_values = list(self.electricity_system_quotas.values()) + list(self.heat_supply_quotas.values()) + values_ct
        return not all(
            _zero_or_hundred(value) for value in all_values
        )

    def get_fixed_and_varying_technologies(self):
        # Building:
        fixed_technologies = []
        varying_technologies = []
        if isinstance(self.construction_type_quota, str):
            fixed_technologies.append(self.construction_type_quota)
        else:
            varying_technologies.append(list(self.construction_type_quota.keys())[0])

        def _append_fixed_and_varying(quotes: dict, fixed: list, varying: list):
            for name, case_value in quotes.items():
                if case_value == 100:
                    fixed.extend(name.split("+"))
                elif case_value > 0:
                    varying.append(name)

        _append_fixed_and_varying(self.heat_supply_quotas, fixed_technologies, varying_technologies)
        _append_fixed_and_varying(self.electricity_system_quotas, fixed_technologies, varying_technologies)
        return fixed_technologies, varying_technologies


@dataclasses.dataclass
class QuotaVariation:
    quota_cases: Dict[str, Quotas]
    fixed_technologies: List[str]
    varying_technologies: Union[List[List[str]], Dict[str, List[int]]]

    def get_varying_technology_ids(self):
        if isinstance(self.varying_technologies, dict):
            return [f"{v}%" for v in self.get_single_varying_technology_name_and_quotas()[1]]
        return ["_".join(l) for l in self.varying_technologies]

    def get_single_varying_technology_name_and_quotas(self):
        if not isinstance(self.varying_technologies, dict):
            raise KeyError("This function is only supported for numeric quotas")
        tech = next(iter(self.varying_technologies))
        return tech, self.varying_technologies[tech]

    def get_quota_case_name_and_value_dict(self):
        return dict(zip(self.quota_cases.keys(), self.get_varying_technology_ids()))

    def pretty_name(self, technology):
        pretty_name_map = {
            "p_adv_ret": "Advanced-retrofit",
            "p_ret": "Retrofit",
        }
        return pretty_name_map.get(technology, technology.capitalize())


def _draw_from_dict(d: dict, n_draws: int):
    options = list(d.keys())
    probabilities = np.array(list(d.values()))
    probabilities = probabilities / sum(probabilities)
    counter = 0
    while counter < 1e6:
        drawn_choices = choices(options, weights=probabilities, k=n_draws)
        count_choices = np.array([drawn_choices.count(option) for option in options])
        resulting_probabilities_match = (
            (np.floor(probabilities * n_draws) <= count_choices) &
            (count_choices <= np.ceil(probabilities * n_draws))
        )
        if np.all(resulting_probabilities_match):
            return drawn_choices
        counter += 1
        if counter > 1e4:
            logger.debug(
                "%s - Draw does not match given probability: %s, %s",
                counter, count_choices, probabilities * n_draws
            )
    raise ValueError("Search took too long, terminating")  # Or return?



def _shuffle_grid_from_dict(d: dict, n_draws: int):
    options = list(d.keys())
    probabilities = np.array(list(d.values()))
    probabilities = probabilities / sum(probabilities)
    import random
    random.shuffle
    drawn_choices = choices(options, weights=probabilities, k=n_draws)
    count_choices = np.array([drawn_choices.count(option) for option in options])
    resulting_probabilities_match = (
        (np.floor(probabilities * n_draws) <= count_choices) &
        (count_choices <= np.ceil(probabilities * n_draws))
    )
    return drawn_choices

def _draw_uncertain_choices(
        df_grid: pd.DataFrame,
        quotas: Quotas,
):
    construction_type_choices = {}
    building_types = df_grid.loc[:, "Gebäudetyp"].apply(lambda x: x.split(" ")[0])  # In case of MFH (..)
    for building_type in building_types.unique():
        df_grid_building_type = df_grid.loc[building_type == building_types]
        for year_of_construction in df_grid_building_type.loc[:, "Baujahr"].unique():
            df_grid_building_type_year = df_grid_building_type.loc[year_of_construction == df_grid_building_type.loc[:, "Baujahr"]]
            _choices_for_building_and_year = _draw_from_dict(
                quotas.construction_type_quotas[building_type][year_of_construction],
                n_draws=len(df_grid_building_type_year)
            )
            for i, house_idx in enumerate(df_grid_building_type_year.index):
                construction_type_choices[house_idx] = _choices_for_building_and_year[i]

    heat_supply_choice = _draw_from_dict(quotas.heat_supply_quotas, n_draws=len(df_grid))
    electricity_system_choice = _draw_from_dict(quotas.electricity_system_quotas, n_draws=len(df_grid))
    all_choices = {}
    for idx, house_idx in enumerate(df_grid.index):
        all_choices[house_idx] = {
            "heat_supply_choice": heat_supply_choice[idx],
            "electricity_system_choice": electricity_system_choice[idx],
            "construction_type_choice": construction_type_choices[house_idx]
        }
    return all_choices


def _get_time_series_data_for_choices(
        heat_supplies: dict,
        house_index: int,
        house_choices: dict,
        df_teaser: pd.DataFrame = None
):
    df_sim, time_series_data = heat_supplies[house_choices["heat_supply_choice"]]
    possible_rows = df_sim.loc[house_index]
    if isinstance(possible_rows, pd.Series):
        if not possible_rows["Baujahr"] == 2010:
            raise KeyError("Only one type even though not 2010, something went wrong")
    else:
        mask = possible_rows.loc[:, "construction_type"] == house_choices["construction_type_choice"]
        if not np.any(mask):
            raise KeyError("No mask fitted, something went wrong")
        possible_rows = possible_rows.loc[mask]
        if len(possible_rows) != 1:
            raise TypeError("Somehting went wrong")
        possible_rows = possible_rows.iloc[0]

    sim_result_name = possible_rows["simulation_result"]
    time_series_data_for_choice = time_series_data[sim_result_name].loc[:, house_choices["electricity_system_choice"]]

    names_to_store_for_plausibility = {
        "building_demand": 1,
        "heat_demand": 1,
        "dhw_demand": 1,
        "ABui": 1,
        "heat_load": 1,
        "SCOP_Sys": 1,
        "WEleGen": 1,
        "PEleMax": 1
    }
    plausibility_check = {name: possible_rows[name] / factor for name, factor in names_to_store_for_plausibility.items()}
    if df_teaser is not None:
        building_name = f'{possible_rows["Baujahr"]}_{possible_rows["construction_type"].replace("tabula_", "")}'
        plausibility_check["heat_load_teaser"] = df_teaser.loc[building_name, "heat_load"]
        plausibility_check["heat_demand_teaser"] = df_teaser.loc[building_name, "heat_demand"]
        plausibility_check["ABui_teaser"] = df_teaser.loc[building_name, "net_leased_area"]

    return time_series_data_for_choice, plausibility_check


def load_function_kwargs_prior_to_monte_carlo(
        hybrid: Path,
        monovalent: Path,
        heating_rod: Path,
        grid_case: str):
    t0 = time.time()

    df_grid = pd.read_excel(
        KERBER_NETZ_XLSX,
        sheet_name=f"Kerber Netz {grid_case}",
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
    df_sim_gas = df_sim.copy()
    df_sim_gas.loc[:, "PEleMax"] = 0
    heat_supplies["gas"] = (df_sim_gas, gas_time_series_data)

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
        # return data.sum() * W_to_Wh  # Also grid feed in
        return data[data > 0].sum() * W_to_Wh  # Also grid feed in

    for quota_name, quotas in quota_cases.items():
        for _ in range(quotas.n_monte_carlo):
            grid, all_choices = run_single_grid_simulation(quotas=quotas, **function_kwargs)
            choices_for_grid[quota_name].append(all_choices)
            for point, data in grid.items():
                if point not in max_data:
                    max_data[point] = {quota_name: [data.max()]}
                    sum_data[point] = {quota_name: [_get_electricity_sum_in_kWh(data)]}
                    # tsd_data[point] = {quota_name: [data]}
                elif quota_name not in max_data[point]:
                    max_data[point][quota_name] = [data.max()]
                    sum_data[point][quota_name] = [_get_electricity_sum_in_kWh(data)]
                    # tsd_data[point][quota_name] = [data]
                else:
                    max_data[point][quota_name].append(data.max())
                    sum_data[point][quota_name].append(_get_electricity_sum_in_kWh(data))
                    # tsd_data[point][quota_name].append(data)
        i += 1
        logger.info("Ran quota case %s of %s", i, len(quota_cases))

    monte_carlo_data = {
        "max": max_data,
        "sum": sum_data,
        # "tsd_data": tsd_data,
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

    all_choices = _draw_uncertain_choices(
        df_grid=df_grid,
        quotas=quotas,
    )
    for house_index, row in df_grid.iterrows():
        time_series_result, _ = _get_time_series_data_for_choices(
            heat_supplies=heat_supplies,
            house_index=house_index,
            house_choices=all_choices[house_index]
        )
        grid[f"AP_{row['Anschlusspunkt'].split('-')[0]}"] += time_series_result
    grid["Trafo"] = np.sum(list(grid.values()), axis=0)  # Build overall sum at ONT
    return grid, all_choices


def get_grid_simulation_input_for_choices(
        heat_supplies: dict,
        df_grid: pd.DataFrame,
        choices_for_grid: dict
):
    grid_time_series_data = []
    grid_plausibility = []
    df_teaser = pd.read_excel(DATA_PATH.joinpath("TEASERComparison.xlsx"), index_col=0)

    for house_index, row in df_grid.iterrows():
        time_series_data, plausibility_data = _get_time_series_data_for_choices(
            heat_supplies=heat_supplies,
            house_index=house_index,
            house_choices=choices_for_grid[house_index],
            df_teaser=df_teaser
        )
        grid_plausibility.append(plausibility_data)
        grid_time_series_data.append(time_series_data)
    return grid_time_series_data, grid_plausibility


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
    return arg_of_value(arr, mean)


def argmedian(arr):
    return arg_of_value(arr, np.median(arr))


def arg_of_value(arr, value):
    # Calculate the absolute differences between each element and the value
    abs_diff = np.abs(arr - value)
    # Find the index of the element with the smallest absolute difference
    return np.argmin(abs_diff)


def arg_percentile(arr, percentile):
    percentile_value = np.percentile(arr, percentile)
    return arg_of_value(arr, percentile_value)


def percentile_5(arr):
    return arg_percentile(arr, 5)


def percentile_95(arr):
    return arg_percentile(arr, 95)


def percentile_03(arr):
    return arg_percentile(arr, 0.3)


def percentile_997(arr):
    return arg_percentile(arr, 99.7)


def get_grid_simulation_case_name(quota_case: str, grid_case: str):
    return quota_case + "_" + grid_case


def plot_and_export_single_monte_carlo(
        quota_variation: QuotaVariation,
        data,
        metric: str,
        arg_function: callable,
        save_path: Path,
        grid_case: str,
        plots_only: bool,
        heat_supplies: dict, df_grid: pd.DataFrame
):
    export_data = {}
    emissions_data = {}
    quota_case_grid_simulation_inputs = {}
    quota_case_grid_data = {}
    simultaneity_factors = {}
    for quota_case in quota_variation.quota_cases:
        arg = arg_function(data[metric]["Trafo"][quota_case])
        # Save in excel for Lastflusssimulation:
        choices_for_grid = data["choices_for_grid"][quota_case][arg]
        grid_time_series_data, grid_plausibility = get_grid_simulation_input_for_choices(
            heat_supplies=heat_supplies,
            df_grid=df_grid,
            choices_for_grid=choices_for_grid,
        )
        df_plausibility = pd.DataFrame(
            grid_plausibility, index=df_grid["Gebäudetyp"].values
        )
        df_plausibility.to_excel(save_path.joinpath(f"grid_plausibility_{quota_case}.xlsx"))
        quota_case_grid_data[quota_case] = grid_time_series_data
        # Plot resulting distribution
        plots.plot_technology_choices_in_grid(
            df_grid=df_grid,
            choices_for_grid=choices_for_grid,
            save_path=save_path.joinpath(f"grid_choices_{quota_case}.png")
        )
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

        max_over_households = 534.9015984368119
        max_peak_per_e_mobility = 11
        n_house_with_e_mobility = len([choice for choice in choices_for_grid.values() if "e_mob" in choice["electricity_system_choice"]])
        trafo_max_possible = (
                df_plausibility.loc[:, "PEleMax"].sum() / 1000 +
                max_over_households +
                n_house_with_e_mobility * max_peak_per_e_mobility
        )
        export_data[quota_case] = {
            "max": {point: data["max"][point][quota_case][arg] for point in data["max"].keys()},
            "sum": {point: data["sum"][point][quota_case][arg] for point in data["sum"].keys()}
        }
        max_trafo = export_data[quota_case]['max']['Trafo']
        simultaneity_factor = max_trafo / trafo_max_possible
        if simultaneity_factor > 1:
            logger.error(f"{simultaneity_factor=} for {quota_case=}")
        simultaneity_factors[quota_case] = {
            "max": max_trafo, "max_possible": trafo_max_possible, "factor": simultaneity_factor
        }

    with open(save_path.joinpath("simultaneity_factors.json"), "w+") as file:
        json.dump(simultaneity_factors, file, indent=2)
    plots.plot_time_series(
        quota_case_grid_data=quota_case_grid_data,
        save_path=save_path,
        quota_variation=quota_variation
    )

    return {"grid": export_data, "grid_simulation_inputs": quota_case_grid_simulation_inputs}


def save_excel(df, path, sheet_name):
    if path.exists():
        with pd.ExcelWriter(path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        df.to_excel(path, sheet_name=sheet_name)


def run_save_and_plot_monte_carlo(
        kwargs: dict
):
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level="INFO")

    quota_variation: QuotaVariation = kwargs["quota_variation"]
    grid_case: str = kwargs["grid_case"]
    save_path: Path = kwargs["save_path"]
    load: bool = kwargs["load"]
    recreate_plots: bool = kwargs["recreate_plots"]
    extra_case_name_hybrid: str = kwargs["extra_case_name_hybrid"]

    # Load separately as large kwargs can't be passed by multiprocessing
    function_kwargs = load_function_kwargs_for_grid(extra_case_name_hybrid=extra_case_name_hybrid, grid_case=grid_case)

    os.makedirs(save_path, exist_ok=True)
    pickle_path = save_path.joinpath(f"monte_carlo_{grid_case}.pickle")
    all_results_path = save_path.joinpath("results_to_plot.json")

    if os.path.exists(all_results_path) and os.path.exists(pickle_path) and not recreate_plots:
        return

    if load and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
        logger.info("Loaded pickle data from %s", pickle_path)
    else:
        data = run_monte_carlo_sim(quota_cases=quota_variation.quota_cases, function_kwargs=function_kwargs)
        with open(pickle_path, "wb") as file:
            pickle.dump(data, file)

    plots.plot_monte_carlo_bars(data=data, metric="max", save_path=save_path, quota_variation=quota_variation)
    plots.plot_monte_carlo_bars(data=data, metric="sum", save_path=save_path, quota_variation=quota_variation)
    plots.plot_monte_carlo_violin(data=data, metric="max", save_path=save_path, quota_variation=quota_variation)
    plots.plot_monte_carlo_violin(data=data, metric="sum", save_path=save_path, quota_variation=quota_variation)
    plots.plot_monte_carlo_convergence(data=data, metric="max", save_path=save_path, quota_variation=quota_variation)
    plots.plot_monte_carlo_convergence(data=data, metric="sum", save_path=save_path, quota_variation=quota_variation)
    export_data = {}
    for arg_function in [
        argmean,
        #np.argmax,
        #np.argmin,
        percentile_03,
        percentile_997,
    ]:
        save_path_arg_function = save_path.joinpath(arg_function.__name__)
        os.makedirs(save_path_arg_function, exist_ok=True)
        export_data[arg_function.__name__] = plot_and_export_single_monte_carlo(
            quota_variation=quota_variation,
            arg_function=arg_function,
            data=data,
            metric="max",
            plots_only=False,
            save_path=save_path_arg_function,
            grid_case=grid_case,
            **function_kwargs
        )

    with open(all_results_path, "w") as file:
        json.dump(export_data, file)
    return True


def _create_quotas_from_0_to_100(
        quota_study_name: str,
        quota_variable: str,
        arg_wrapper: callable = None,
        zero_to_hundred: list = None,
        **quota_kwargs
):
    if arg_wrapper is None:
        arg_wrapper = lambda x: x
    if zero_to_hundred is None:
        zero_to_hundred = [0, 20, 40, 60, 80, 100]
    quota_cases = {}
    for quota in zero_to_hundred:
        quota_value = arg_wrapper(quota)
        quota_cases[f"{quota_variable}_{quota_study_name}_{quota}"] = Quotas(
            **{
                quota_variable: quota_value,
                **quota_kwargs
            }
        )
    fixed_technologies = []
    for tech, value in quota_kwargs.items():
        if value == 100:
            if tech == "pv_battery_quota":
                fixed_technologies.extend(["pv", "battery"])
            else:
                fixed_technologies.append(tech.replace("_quota", ""))
        if tech == "heat_pump_quota" and value == 0:
            fixed_technologies.append("gas")
        if tech == "construction_type_quota":
            fixed_technologies.append(value)

    if isinstance(zero_to_hundred[0], str):
        varying_technologies = [[z] for z in zero_to_hundred]
    else:
        # Numeric changes, fixed technology type:
        if isinstance(quota_value, dict):
            varying_technology_clean_name = next(iter(quota_value))
        else:
            varying_technology_clean_name = quota_variable.replace("_quota", "")
        varying_technologies = {varying_technology_clean_name: zero_to_hundred}

    return QuotaVariation(
        quota_cases=quota_cases,
        fixed_technologies=fixed_technologies,
        varying_technologies=varying_technologies
    )


def add_studies_to_compare_to_old_results(all_quota_studies):
    all_quota_studies["CompareOldAndNew_average"] = QuotaVariation(quota_cases={
        "Hybrid": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=0, hybrid_quota=100, heating_rod_quota=0
        ),
        "Monovalent": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
        )},
        fixed_technologies=[],
        varying_technologies=[
            ["average", "hybrid"],
            ["average", "heat_pump"],
        ]
    )
    all_quota_studies["CompareOldAndNew_HR_average"] = QuotaVariation(quota_cases={
        "Hybrid": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=0, hybrid_quota=100, heating_rod_quota=0
        ),
        "Monovalent": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )},
        fixed_technologies=[],
        varying_technologies=[
            ["average", "hybrid"],
            ["average", "heating_rod"],
        ]
    )
    all_quota_studies["CompareOldAndNew_HR_no_retrofit"] = QuotaVariation(quota_cases={
        "Hybrid": Quotas(
            construction_type_quota="no_retrofit", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=0, hybrid_quota=100, heating_rod_quota=0
        ),
        "Monovalent": Quotas(
            construction_type_quota="no_retrofit", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=50, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )},
        fixed_technologies=[],
        varying_technologies=[
            ["no_retrofit", "hybrid"],
            ["no_retrofit", "heat_pump"],
        ]
    )


def add_graphical_abstract_study(all_quota_studies):
    all_quota_studies["GraphicalAbstract"] = QuotaVariation(quota_cases={
        "GraphicalAbstract_GasAverage": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=0, hybrid_quota=0, heating_rod_quota=0
        ),
        "GraphicalAbstract_HybAverage": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=100, heating_rod_quota=0
        ),
        "GraphicalAbstract_MonAverage": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=0
        ),
        "GraphicalAbstract_HeaRodAverage": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=0, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        ),
        "GraphicalAbstract_HeaRodAverageEMob": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=0,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        ),
        "GraphicalAbstract_HeaRodAverageEMobPV": Quotas(
            construction_type_quota="average", pv_quota=100, pv_battery_quota=0,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        ),
        "GraphicalAbstract_HeaRodAverageEMobPVBat": Quotas(
            construction_type_quota="average", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        ),
        "GraphicalAbstract_HeaRodAllRetroEMobPVBat": Quotas(
            construction_type_quota="all_retrofit", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        ),
        "GraphicalAbstract_HeaRodAllAdvRetroEMobPVBat": Quotas(
            construction_type_quota="all_adv_retrofit", pv_quota=0, pv_battery_quota=100,
            e_mobility_quota=100, heat_pump_quota=100, hybrid_quota=0, heating_rod_quota=100
        )},
        fixed_technologies=[],
        varying_technologies=[
            ["average", "gas"],
            ["average", "hybrid"],
            ["average", "heat_pump"],
            ["average", "heating_rod"],
            ["average", "heating_rod", "e_mobility"],
            ["average", "heating_rod", "e_mobility", "pv"],
            ["average", "heating_rod", "e_mobility", "pv", "battery"],
            ["all_retrofit", "heating_rod", "e_mobility", "pv", "battery"],
            ["all_adv_retrofit", "heating_rod", "e_mobility", "pv", "battery"],
        ]
    )


def add_single_analysis_study(all_quota_studies):
    all_quota_studies["AnalyseHP"] = _create_quotas_from_0_to_100(
        quota_study_name="AnalyseHP",
        quota_variable="heat_pump_quota",
        construction_type_quota="average",
        e_mobility_quota=0,
        pv_battery_quota=0,
        hybrid_quota=0,
        heating_rod_quota=0,
        pv_quota=0
    )
    all_quota_studies["AnalyseHR"] = _create_quotas_from_0_to_100(
        quota_study_name="AnalyseHR",
        quota_variable="heating_rod_quota",
        construction_type_quota="average",
        e_mobility_quota=0,
        pv_battery_quota=0,
        hybrid_quota=0,
        heat_pump_quota=100,
        pv_quota=0
    )
    all_quota_studies["AnalyseEMobility"] = _create_quotas_from_0_to_100(
        quota_study_name="AnalyseEMobility",
        quota_variable="e_mobility_quota",
        construction_type_quota="average",
        pv_quota=0,
        pv_battery_quota=0,
        hybrid_quota=0,
        heat_pump_quota=100,
        heating_rod_quota=100
    )
    all_quota_studies["AnalysePV"] = _create_quotas_from_0_to_100(
        quota_study_name="AnalysePV",
        quota_variable="pv_quota",
        construction_type_quota="average",
        e_mobility_quota=100,
        pv_battery_quota=0,
        hybrid_quota=0,
        heat_pump_quota=100,
        heating_rod_quota=100
    )
    all_quota_studies["AnalysePVBat"] = _create_quotas_from_0_to_100(
        quota_study_name="AnalysePVBat",
        quota_variable="pv_battery_quota",
        construction_type_quota="average",
        e_mobility_quota=100,
        pv_quota=0,
        hybrid_quota=0,
        heat_pump_quota=100,
        heating_rod_quota=100
    )


def get_all_quota_studies():
    all_quota_studies = {}
    fixed_pv_quotas = [0, 100]
    fixed_pv_battery_quotas = [0, 100]
    fixed_e_mobility_quotas = [0, 100]
    fixed_heat_pump_quotas = [100]
    fixed_heating_rod_quotas = [0, 100]
    fixed_hybrid_quotas = [0, 100]

    # Gap-1: Hybrid
    for pv, pv_bat, e_mob, hp, hr in itertools.product(
            fixed_pv_quotas,
            fixed_pv_battery_quotas,
            fixed_e_mobility_quotas,
            fixed_heat_pump_quotas,
            fixed_heating_rod_quotas,
    ):
        if pv == 100 and pv_bat == 100:
            continue
        tech_values = {
            "PV": pv,
            "PVBat": pv_bat,
            "EMob": e_mob,
            "HP": hp,
            "HR": hr,
        }
        identifier = "_".join([tech for tech, value in tech_values.items() if value == 100])
        all_quota_studies[f"hybrid_{identifier}"] = _create_quotas_from_0_to_100(
            quota_study_name=identifier,
            quota_variable="hybrid_quota",
            construction_type_quota="average",
            pv_quota=pv,
            pv_battery_quota=pv_bat,
            e_mobility_quota=e_mob,
            heat_pump_quota=hp,
            heating_rod_quota=hr
        )

    # Gap-2: Building retrofit
    for pv_bat, e_mob, hp, hr, hyb in itertools.product(
            fixed_pv_battery_quotas,
            fixed_e_mobility_quotas,
            fixed_heat_pump_quotas,
            fixed_heating_rod_quotas,
            fixed_hybrid_quotas
    ):
        if hyb == 100 and hr == 100:
            continue  # Makes no difference

        tech_values = {
            "PVBat": pv_bat,
            "EMob": e_mob,
            "HP": hp,
            "HR": hr,
            "Hyb": hyb
        }
        identifier = "_".join([tech for tech, value in tech_values.items() if value == 100])
        kwargs = dict(
            quota_study_name=identifier,
            quota_variable="construction_type_quota",
            pv_quota=0,
            pv_battery_quota=pv_bat,
            e_mobility_quota=e_mob,
            heat_pump_quota=hp,
            heating_rod_quota=hr,
            hybrid_quota=hyb
        )
        #all_quota_studies[f"extrema_{identifier}"] = _create_quotas_from_0_to_100(
        #    zero_to_hundred=["no_retrofit", "all_retrofit", "all_adv_retrofit"], **kwargs
        #)
        all_quota_studies[f"retrofit_{identifier}"] = _create_quotas_from_0_to_100(
            arg_wrapper=lambda x: dict(p_ret=x / 100), **kwargs
        )
        all_quota_studies[f"adv_retrofit_{identifier}"] = _create_quotas_from_0_to_100(
            arg_wrapper=lambda x: dict(p_adv_ret=x / 100), **kwargs
        )

    add_single_analysis_study(all_quota_studies)
    #add_graphical_abstract_study(all_quota_studies)
    #add_studies_to_compare_to_old_results(all_quota_studies)

    for folder in os.listdir(RESULTS_MONTE_CARLO_FOLDER):
        path = RESULTS_MONTE_CARLO_FOLDER.joinpath(folder)
        if (
                folder not in
                [f"oldbuildings_{k}" for k in all_quota_studies.keys()] +
                [f"newbuildings_{k}" for k in all_quota_studies.keys()] and
                os.path.isdir(path)
        ):
            print(f"Folder {path} could be deleted, not relevant with current factorial design.")
            #shutil.rmtree(path)

    return all_quota_studies


def load_function_kwargs_for_grid(extra_case_name_hybrid: str, grid_case: str, recreate_pickle=False):
    pickle_path = RESULTS_MONTE_CARLO_FOLDER.joinpath(f"{extra_case_name_hybrid}_{grid_case}.pickle")
    if os.path.exists(pickle_path) and not recreate_pickle:
        with open(pickle_path, "rb") as file:
            return pickle.load(file)

    function_kwargs = load_function_kwargs_prior_to_monte_carlo(
        hybrid=RESULTS_BES_FOLDER.joinpath(f"Hybrid{extra_case_name_hybrid}_{grid_case}"),
        monovalent=RESULTS_BES_FOLDER.joinpath(f"Monovalent{extra_case_name_hybrid}_{grid_case}"),
        heating_rod=RESULTS_BES_FOLDER.joinpath(f"Monovalent{extra_case_name_hybrid}_{grid_case}_HR"),
        grid_case=grid_case
    )
    with open(pickle_path, "wb") as file:
        pickle.dump(function_kwargs, file)
    return function_kwargs


def run_all_cases(grid_case: str, load: bool, extra_case_name_hybrid: str = "", n_cpu: int = 1, recreate_plots: bool = True):
    all_quota_cases = get_all_quota_studies()

    multiprocessing_function_kwargs = []
    # Trigger generation of pickle for inputs
    load_function_kwargs_for_grid(extra_case_name_hybrid=extra_case_name_hybrid, grid_case=grid_case, recreate_pickle=False)
    for quota_study_name, quota_variation in all_quota_cases.items():
        save_path = RESULTS_MONTE_CARLO_FOLDER.joinpath(f"{grid_case}_{quota_study_name}")
        multiprocessing_function_kwargs.append(dict(
            quota_variation=quota_variation,
            grid_case=grid_case,
            save_path=save_path,
            load=load,
            recreate_plots=recreate_plots,
            extra_case_name_hybrid=extra_case_name_hybrid,
        ))
    if n_cpu > 1:
        pool = multiprocessing.Pool(processes=n_cpu)
        i = 0
        for res in pool.imap(run_save_and_plot_monte_carlo, multiprocessing_function_kwargs):
            logger.info("Calculated %s of %s cases", i + 1, len(multiprocessing_function_kwargs))
            i += 1
    else:
        for i, kwargs in enumerate(multiprocessing_function_kwargs):
            try:
                run_save_and_plot_monte_carlo(kwargs)
            except Exception as err:
                raise err
                logger.error("Could not calculate case %s: %s", i + 1, err)
            logger.info("Calculated %s of %s cases", i + 1, len(multiprocessing_function_kwargs))


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    PlotConfig.load_default()  # Trigger rc_params
    #run_all_cases(grid_case="newbuildings", load=True, extra_case_name_hybrid="Weather", n_cpu=20)
    run_all_cases(grid_case="oldbuildings", load=True, extra_case_name_hybrid="Weather", n_cpu=20)
