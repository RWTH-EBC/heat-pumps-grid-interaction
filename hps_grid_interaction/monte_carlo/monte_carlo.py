import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict
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
            heat_pump_quota: int,
            hybrid_quota: int,
            pv_quotas: int,
            pv_battery_quotas: int,
            e_auto_quotas: int,
            n_monte_carlo: int = 1000
    ):
        self.construction_type_quotas = get_construction_type_quotas(assumption=construction_type_quota)
        self.heat_supply_quotas = {
            "monovalent": (100 - hybrid_quota) * heat_pump_quota,
            "hybrid": hybrid_quota * heat_pump_quota,
            "gas": (100 - heat_pump_quota)
        }
        self.pv_quotas = {}
        self.pv_battery_quotas = {}
        self.e_auto_quotas = {}
        self.n_monte_carlo = n_monte_carlo


def _draw_uncertain_choice(quotas: Quotas, building_type: str, year_of_construction: int, possible_rows):

    def _draw_from_dict(d: dict):
        return choices(list(d.keys()), list(d.values()), k=1)[0]

    construction_type_choice = _draw_from_dict(quotas.construction_type_quotas[building_type][year_of_construction])
    heat_supply_choice = _draw_from_dict(quotas.heat_supply_quotas)
    pv_choice = _draw_from_dict(quotas.pv_quotas)
    pv_battery_choice = _draw_from_dict(quotas.pv_battery_quotas)
    e_auto_choice = _draw_from_dict(quotas.e_auto_quotas)

    mask = (
            (possible_rows.loc[:, "construction_type"] == construction_type_choice) &
            (possible_rows.loc[:, "heat_supply"] == heat_supply_choice) &
            (possible_rows.loc[:, "pv"] == pv_choice) &
            (possible_rows.loc[:, "pv_battery"] == pv_battery_choice) &
            (possible_rows.loc[:, "e_auto"] == e_auto_choice)
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

    def _load_csv_data(path: Path):
        data = {}
        ordered_sim_results = {}
        for file in os.listdir(path.joinpath("csv_files")):
            if file.endswith(".csv"):
                df = pd.read_csv(path.joinpath("csv_files", file)).set_index("Time")
                assert len(df.columns) == 1, "Only one column is expected"
                file_path = str(path.joinpath("csv_files", file).absolute())
                data[file_path] = df.loc[:, df.columns[0]].values
                n_sim = int(file.split("_")[0])
                ordered_sim_results[n_sim] = file_path
        ordered_sim_results = {key: ordered_sim_results[key] for key in sorted(ordered_sim_results)}
        return data, ordered_sim_results

    time_series_data, sim_results = _load_csv_data(hybrid)
    df_sim_hyb.loc[:, "simulation_result"] = list(sim_results.values())
    df_sim_hyb.loc[:, "system_type"] = "hybrid"
    time_series_data_m, sim_results_mon = _load_csv_data(monovalent)
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


def run_monte_carlo_sim(quota_cases: Dict[str, Quotas], function_kwargs: dict):
    max_data = {}
    sum_data = {}
    tsd_data = {}
    monte_carlo_history = {"Hybrid": [], "Monovalent": []}
    i = 0
    for quota_name, quotas in quota_cases.items():
        for _ in range(quotas.n_monte_carlo):
            grid, _, retrofit_distribution = run_single_grid_simulation(quotas=quotas, **function_kwargs)
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
        df_sim: pd.DataFrame,
        df_grid: pd.DataFrame,
        quotas: Quotas,
        time_series_data: dict
):
    grid = {key: 0 for key in [f"AP_{i + 1}" for i in range(10)]}
    csv_files_for_acs = []
    _resulting_retrofit_distribution = []
    for house_index, row in df_grid.iterrows():
        building_type = row["Gebäudetyp"].split(" ")[0]  # In case of MFH (..)
        sim_result_name = _draw_uncertain_choice(
            quotas=quotas,
            building_type=building_type,
            year_of_construction=row["Baujahr"],
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


def get_grid_simulation_case_name(quota_case: str, case_name: str):
    return quota_case + "_" + case_name


def plot_and_export_single_monte_carlo(
        quota_cases: List[str],
        data, metric: str, save_path: Path,
        case_name: str, grid_case: str, df_sim: pd.DataFrame
):
    arg_function = argmean
    export_data = {}
    emissions_data = {}
    for quota_case in ["Hybrid", "Monovalent"]:
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
        save_excel(df=pd.DataFrame(export_data[quota_case]), path=save_path.parent.joinpath(f"MonteCarloResults.xlsx"),
                   sheet_name=f"{metric}_{short_sheet_name}")
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
    if not hybrid_path.exists():
        logger.error(f"Did not find {hybrid_path}. Skipping case {case_name}")
        return res
    kwargs = load_function_kwargs_prior_to_monte_carlo(
        hybrid=hybrid_path,
        monovalent=RESULTS_BES_FOLDER.joinpath(f"Monovalent_{grid_case}{with_hr_str}"),
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
        "all_retrofit": Quotas(construction_type_quota="all_retrofit"),
        "all_adv_retrofit": Quotas(construction_type_quota="all_adv_retrofit"),
        "no_retrofit": Quotas(construction_type_quota="no_retrofit")
    }

    res = {}
    for grid_case in ["altbau", "neubau"]:
        for with_hr in [True, False]:
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
    PATH = run_all_cases(load=True, extra_case_name_hybrid="GEGBiv")
