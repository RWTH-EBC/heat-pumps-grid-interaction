import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List
from random import choices

import pandas as pd
import numpy as np

from hps_grid_interaction.plotting.config import PlotConfig
from hps_grid_interaction.utils import get_construction_type_quotas
from hps_grid_interaction.emissions import COLUMNS_EMISSIONS
from hps_grid_interaction.monte_carlo import plots
from hps_grid_interaction import RESULTS_BES_FOLDER
from hps_grid_interaction.bes_simulation.simulation import W_to_Wh
logger = logging.getLogger(__name__)


COLUMNS_GEG = ["percent_renewables", "QRenewable", "QBoi"]


class Quotas:

    def __init__(
            self,
            construction_type_quota: str,
            heat_pump_quota: int = 100,
            hybrid_quota: int = 0,
            pv_quota: int = 100,
            pv_battery_quota: int = 100,
            e_mobility_quota: int = 0,
            n_monte_carlo: int = 1000
    ):
        self.construction_type_quotas = get_construction_type_quotas(assumption=construction_type_quota)
        self.heat_supply_quotas = {
            "monovalent": (100 - hybrid_quota) * heat_pump_quota / 100,
            "hybrid": hybrid_quota * heat_pump_quota / 100,
            "gas": (100 - heat_pump_quota)
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


def _draw_uncertain_choice(quotas: Quotas, building_type: str, year_of_construction: int, heat_supply: dict, house_index: int):

    def _draw_from_dict(d: dict):
        return choices(list(d.keys()), list(d.values()), k=1)[0]

    construction_type_choice = _draw_from_dict(quotas.construction_type_quotas[building_type][year_of_construction])
    heat_supply_choice = _draw_from_dict(quotas.heat_supply_quotas)
    electricity_system_choice = _draw_from_dict(quotas.electricity_system_quotas)

    df_sim, time_series_data = heat_supply[heat_supply_choice]
    possible_rows = df_sim.loc[house_index]
    mask = possible_rows.loc[:, "construction_type"] == construction_type_choice
    if not np.any(mask):
        raise KeyError("No mask fitted, something went wrong")
    sim_result_name = possible_rows.loc[mask, "simulation_result"].values[0]

    tsd = time_series_data[sim_result_name].loc[:, electricity_system_choice]

    all_choices = {
        "heat_supply_choice": heat_supply_choice,
        "electricity_system_choice": electricity_system_choice,
        "construction_type_choice": construction_type_choice
    }

    return tsd, all_choices


def load_function_kwargs_prior_to_monte_carlo(
        hybrid: Path,
        monovalent: Path,
        hybrid_e_mobility: Path,
        monovalent_e_mobility: Path,
        grid_case: str):

    def _load_csv_data(path: Path):
        data = {}
        ordered_sim_results = {}
        for file in os.listdir(path.joinpath("csv_files")):
            if file.endswith(".csv"):
                file_path = str(path.joinpath("csv_files", file).absolute())
                df = pd.read_csv(file_path).set_index("Time")
                data[file_path] = df
                n_sim = int(file.split("_")[0])
                ordered_sim_results[n_sim] = file_path
        ordered_sim_results = {key: ordered_sim_results[key] for key in sorted(ordered_sim_results)}
        return data, list(ordered_sim_results.values())
    cases = {
        "hybrid": {"e_mobility": hybrid_e_mobility, "no": hybrid},
        "monovalent": {"e_mobility": monovalent_e_mobility, "no": monovalent}
    }
    heat_supplys = {}
    for heat_supply, paths in cases.items():
        # Should not matter which xlsx is loaded, with or without e-mobility
        df_sim = pd.read_excel(
            paths["no"].joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
            sheet_name="Sheet1",
            index_col=0
        )
        time_series_data, simulation_result = _load_csv_data(paths["no"])
        df_sim.loc[:, "simulation_result"] = simulation_result
        time_series_data_e_mobility, _ = _load_csv_data(paths["e_mobility"])
        time_series_data = pd.concat([time_series_data, time_series_data_e_mobility])
        heat_supplys[heat_supply] = (df_sim, time_series_data)

    heat_supplys["gas"] = 0

    df_grid = pd.read_excel(
        Path(__file__).parent.joinpath("Kerber_Vorstadtnetz.xlsx"),
        sheet_name=f"Kerber Netz {grid_case.capitalize()}",
        index_col=0
    )
    func_kwargs = dict(heat_supplys=heat_supplys, df_grid=df_grid)
    return func_kwargs


def run_monte_carlo_sim(quota_cases: Dict[str, Quotas], function_kwargs: dict):
    max_data = {}
    sum_data = {}
    tsd_data = {}
    monte_carlo_history = {"Hybrid": [], "Monovalent": []}
    i = 0
    for quota_name, quotas in quota_cases.items():
        for _ in range(quotas.n_monte_carlo):
            grid, retrofit_distribution = run_single_grid_simulation(quotas=quotas, **function_kwargs)
            monte_carlo_history[quota_name].append(retrofit_distribution)
            for point, data in grid.items():
                if point not in max_data:
                    max_data[point] = {quota_name: [data.max()]}
                    sum_data[point] = {quota_name: [data.sum() * W_to_Wh]}
                    #tsd_data[point] = {quota_name: [data]}
                elif quota_name not in max_data[point]:
                    max_data[point][quota_name] = [data.max()]
                    sum_data[point][quota_name] = [data.sum() * W_to_Wh]
                    #tsd_data[point][quota_name] = [data]
                else:
                    max_data[point][quota_name].append(data.max())
                    sum_data[point][quota_name].append(data.sum() * W_to_Wh)
                    #tsd_data[point][quota_name].append(data)
        i += 1
        logger.info("Ran quota case %s of %s", i, len(quota_cases))

    monte_carlo_data = {
        "max": max_data,
        "sum": sum_data,
        #"tsd_data": tsd_data,
        "simulations": monte_carlo_history
    }

    return monte_carlo_data


def run_single_grid_simulation(
        heat_supply: dict,
        df_grid: pd.DataFrame,
        quotas: Quotas,
):
    grid = {key: 0 for key in [f"AP_{i + 1}" for i in range(10)]}

    _resulting_choice_distribution = []
    for house_index, row in df_grid.iterrows():
        building_type = row["Gebäudetyp"].split(" ")[0]  # In case of MFH (..)
        time_series_result, all_choices = _draw_uncertain_choice(
            quotas=quotas,
            building_type=building_type,
            year_of_construction=row["Baujahr"],
            heat_supply=heat_supply,
            house_index=house_index
        )
        _resulting_choice_distribution.append(all_choices)
        grid[f"AP_{row['Anschlusspunkt'].split('-')[0]}"] += time_series_result

    grid["ONT"] = np.sum(list(grid.values()), axis=0)  # Build overall sum at ONT
    return grid, _resulting_choice_distribution


def argmean(arr):
    # Calculate the mean of the array
    mean = np.mean(arr)
    # Calculate the absolute differences between each element and the mean
    abs_diff = np.abs(arr - mean)
    # Find the index of the element with the smallest absolute difference
    return np.argmin(abs_diff)


def get_grid_simulation_case_name(quota_case: str, case_name: str):
    return quota_case + "_" + case_name


def plot_and_export_single_monte_carlo(
        quota_cases: List[str],
        data, metric: str, save_path: Path,
        case_name: str, grid_case: str
):
    arg_function = argmean
    export_data = {}
    emissions_data = {}
    for quota_case in quota_cases:
        arg = arg_function(data[metric]["ONT"][quota_case])
        # Save in excel for Lastflusssimulation:
        to_grid_simulation = data["simulations"][quota_case][arg]

        df_lastfluss = pd.read_excel(
            Path(__file__).parent.joinpath("Kerber_Vorstadtnetz.xlsx"),
            sheet_name=f"{grid_case}_lastfluss_template",
            index_col=0
        )
        df_lastfluss["Wärmepumpenstrom-Zeitreihe"] = to_grid_simulation
        grid_simulation_case_name = get_grid_simulation_case_name(quota_case=quota_case, case_name=case_name)
        workbook_name = save_path.parent.joinpath(f"{grid_simulation_case_name}.xlsx")
        save_excel(df=df_lastfluss, path=workbook_name, sheet_name="lastfluss")
        export_data[quota_case] = {
            "max": {point: data["max"][point][quota_case][arg] for point in data["max"].keys()},
            "sum": {point: data["sum"][point][quota_case][arg] for point in data["sum"].keys()}
        }
        plots.plot_time_series(data=data, quota_case=quota_case, metric=metric, save_path=save_path)

        sim_results = data["simulations"][quota_case][arg]
        # TODO: Fix simulation results for cases
        quota_case_mask = df_sim.loc[:, "system_type"] == tech.lower()
        df_quota_case = df_sim.loc[quota_case_mask]
        columns = COLUMNS_EMISSIONS + COLUMNS_GEG
        sum_cols = {col: 0 for col in columns}
        for sim_result in sim_results:
            row = df_quota_case.loc[df_quota_case.loc[:, "simulation_result"] == sim_result]
            for col in columns:
                sum_cols[col] += row[col].values[0]
        emissions_data[quota_case] = sum_cols
    return {"grid": export_data, "emissions": emissions_data}


def save_excel(df, path, sheet_name):
    if path.exists():
        with pd.ExcelWriter(path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        df.to_excel(path, sheet_name=sheet_name)


def run_save_and_plot_monte_carlo(
        quota_cases: Dict[str, Quotas],
        grid_case: str, with_hr: bool, res: dict,
        load: bool = False,
        extra_case_name: str = ""
):
    all_results = res

    with_hr_str = "_HR" if with_hr else ""
    case_name = f"{grid_case}{with_hr_str}"

    save_path = RESULTS_BES_FOLDER.joinpath(f"MonteCarloResults_{case_name}")
    pickle_path = save_path.joinpath(f"monte_carlo_{case_name}.pickle")

    hybrid_path = RESULTS_BES_FOLDER.joinpath(f"Hybrid{extra_case_name}_{grid_case}")
    hybrid_path_e_mobility = RESULTS_BES_FOLDER.joinpath(f"Hybrid{extra_case_name}_{grid_case}_EMob")

    kwargs = load_function_kwargs_prior_to_monte_carlo(
        hybrid=hybrid_path,
        monovalent=RESULTS_BES_FOLDER.joinpath(f"Monovalent_{grid_case}{with_hr_str}"),
        hybrid_e_mobility=hybrid_path_e_mobility,
        monovalent_e_mobility=RESULTS_BES_FOLDER.joinpath(f"Monovalent_{grid_case}{with_hr_str}_EMob"),
        grid_case=grid_case
    )

    if load and pickle_path.exists():
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
    else:
        import time
        t0 = time.time()
        data = run_monte_carlo_sim(quota_cases=quota_cases, function_kwargs=kwargs)
        logger.info(f"Simulations took {time.time() - t0} s")

    os.makedirs(save_path, exist_ok=True)
    plots.plot_monte_carlo_bars(data=data, metric="max", save_path=save_path)
    plots.plot_monte_carlo_bars(data=data, metric="sum", save_path=save_path)
    plots.plot_monte_carlo_violin(data=data, metric="max", save_path=save_path)
    plots.plot_monte_carlo_violin(data=data, metric="sum", save_path=save_path)
    export_data = plot_and_export_single_monte_carlo(
        quota_cases=quota_cases,
        data=data, metric="max", df_sim=kwargs["df_sim"],
        save_path=save_path, case_name=case_name, grid_case=grid_case
    )
    all_results[case_name] = export_data
    with open(pickle_path, "wb") as file:
        pickle.dump(data, file)
    return all_results


def run_all_cases(load: bool, extra_case_name_hybrid: str = ""):
    quota_cases = {
        "average": Quotas(construction_type_quota="average"),
        #"all_retrofit": Quotas(construction_type_quota="all_retrofit"),
        #"all_adv_retrofit": Quotas(construction_type_quota="all_adv_retrofit"),
        #"no_retrofit": Quotas(construction_type_quota="no_retrofit")
    }

    res = {}
    for grid_case in ["altbau"]:#, "neubau"]:
        for with_hr in [True]:#, False]:
            res = run_save_and_plot_monte_carlo(
                quota_cases=quota_cases,
                grid_case=grid_case,
                with_hr=with_hr, res=res, load=load,
                extra_case_name=extra_case_name_hybrid
            )
    all_results_path = RESULTS_BES_FOLDER.joinpath("results_to_plot.json")
    with open(all_results_path, "w") as file:
        json.dump(res, file)
    return all_results_path


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    PlotConfig.load_default()  # Trigger rc_params
    PATH = run_all_cases(load=True, extra_case_name_hybrid="PVBat")
