import logging
import os
import shutil
from typing import Union

from dataclasses import dataclass

import pandas as pd
import numpy as np

from ebcpy import TimeSeriesData
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt
from pathlib import Path


from hps_grid_interaction.bes_simulation.building import BuildingConfig
from hps_grid_interaction import PROJECT_FOLDER, KERBER_NETZ_XLSX, E_MOBILITY_DATA, HOUSEHOLD_DATA, DATA_PATH
from hps_grid_interaction.bes_simulation.simulation import TIME_STEP

logger = logging.getLogger(__name__)


@dataclass
class HybridSystemAssumptions:
    method: str
    price_electricity: float = 0.3496
    price_gas: float = 0.0934
    emissions_natural_gas: float = 205
    emissions_electricity: Union[float, str] = 474

    # https://www.iwu.de/fileadmin/publikationen/gebaeudebestand/2018_IWU_CischinskyEtDiefenbach_Datenerhebung-Wohngeb%C3%A4udebestand-2016.pdf

    def get_minimum_cop(self):
        assert self.method in ["costs", "emissions"]
        if self.method == "costs":
            return self.price_electricity / self.price_gas
        if isinstance(self.emissions_electricity, str):
            raise NotImplementedError
        else:
            emissions_electricity = self.emissions_electricity
        return emissions_electricity / self.emissions_natural_gas


def get_construction_type_quotas(assumption: str):

    def _get_ct_quotas(tabula_standard, tabula_retrofit, tabula_adv_retrofit):
        assert tabula_standard + tabula_retrofit + tabula_adv_retrofit == 100, "Quotas do not match 100 % in total"
        d = {"tabula_standard": tabula_standard, "tabula_retrofit": tabula_retrofit, "tabula_adv_retrofit": tabula_adv_retrofit}
        q = {
            2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
            1980: d, 1970: d, 1960: d, 1950: d,
        }
        return {"MFH": q, "EFH": q}

    def _retrofit_percentage_of_stock(p_ret: float = 0, p_adv_ret: float = 0):
        return {
            "MFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": (17 + 75) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (17 + 75) * p_ret, "tabula_adv_retrofit": 8 + (17 + 75) * p_adv_ret},
                1970: {"tabula_standard": (35 + 29) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (35 + 29) * p_ret, "tabula_adv_retrofit": 36 + (35 + 29) * p_adv_ret},
                1960: {"tabula_standard": (35 + 29) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (35 + 29) * p_ret, "tabula_adv_retrofit": 36 + (35 + 29) * p_adv_ret},
                1950: {"tabula_standard": (35 + 29) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (35 + 29) * p_ret, "tabula_adv_retrofit": 36 + (35 + 29) * p_adv_ret},
            },
            "EFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": (29 + 65) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (29 + 65) * p_ret, "tabula_adv_retrofit": 6 + (29 + 65) * p_adv_ret},
                1970: {"tabula_standard": (45 + 35) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (45 + 35) * p_ret, "tabula_adv_retrofit": 20 + (45 + 35) * p_adv_ret},
                1960: {"tabula_standard": (45 + 35) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (45 + 35) * p_ret, "tabula_adv_retrofit": 20 + (45 + 35) * p_adv_ret},
                1950: {"tabula_standard": (45 + 35) * (1 - p_ret - p_adv_ret), "tabula_retrofit": (45 + 35) * p_ret, "tabula_adv_retrofit": 20 + (45 + 35) * p_adv_ret}
            }
        }
    if assumption == "average":
        return _retrofit_percentage_of_stock(p_ret=0, p_adv_ret=0)
    elif assumption == "no_retrofit":
        return _get_ct_quotas(tabula_standard=100, tabula_retrofit=0, tabula_adv_retrofit=0)
    elif assumption == "all_adv_retrofit":
        return _get_ct_quotas(tabula_standard=0, tabula_retrofit=0, tabula_adv_retrofit=100)
    elif assumption == "all_retrofit":
        return _get_ct_quotas(tabula_standard=0, tabula_retrofit=100, tabula_adv_retrofit=0)
    elif isinstance(assumption, dict):
        return _retrofit_percentage_of_stock(**assumption)
    raise KeyError(f"Assumption {assumption} not supported")


def load_outdoor_air_temperature():
    df_oda = pd.read_csv(DATA_PATH.joinpath("t_oda.csv"), index_col=0)
    df_oda.index -= df_oda.index[0]
    df_oda.index /= 3600
    return df_oda


def get_additional_packages(buildings):
    """
    Get new packages like buildings into Dymola
    """
    packages = []
    for bui in buildings:
        packages.append(bui.package_path)
    return list(set(packages))


def create_input_for_gains(csv_path: Path, hybrid_assumptions: HybridSystemAssumptions):
    tsd = TimeSeriesData(csv_path, sep=",")
    tsd.to_float_index()
    # TEASER uses 0.75 for machines and 0.5 for lights. As the electricity profiles
    # do not specify the type, we assume the average of both.
    fac_conv = (0.5 + 0.75) / 2
    tsd.loc[:, "rad"] = tsd.loc[:, ("Wirkleistung [kW]", "raw")] * (1 - fac_conv) * 1000  # to W
    tsd.loc[:, "conv"] = tsd.loc[:, ("Wirkleistung [kW]", "raw")] * fac_conv * 1000  # to W
    tsd.loc[:, "COPMin"] = hybrid_assumptions.get_minimum_cop()
    return tsd


def load_buildings_and_gains(
        sheet_name: str, hybrid_assumptions: HybridSystemAssumptions, study_path: Path,
        with_e_mobility: bool,
        with_night_set_back: bool,
        non_optimal_heating_curve: bool,
        with_smart_thermostat: bool):
    logger.info("Loading grid and building data")
    df = pd.read_excel(KERBER_NETZ_XLSX, sheet_name=sheet_name, index_col=0)

    user_modifiers = []
    building_configs = []
    dhw_profiles = []
    dhw_base_path = PROJECT_FOLDER.joinpath("dhw_tappings")

    tabula_areas_sfh = {
        2010: 187,
        1980: 216,
        1950: 111,
        1960: 121,
        1970: 173
    }
    areas_mfh = {
        2010: 84.5,
        1980: 75.1,
        1970: 73.06,
        1960: 67.2,
        1950: 62.4
    }
    flats_per_floor = 2
    dhw_litre_per_person_per_day = 25
    excel_for_clustering = []

    df_users = pd.read_excel(PROJECT_FOLDER.joinpath("Night_set_backs.xlsx"), sheet_name="users")
    for idx, row in df.iterrows():
        building_data = row.to_dict()
        tsd = create_input_for_gains(
            csv_path=HOUSEHOLD_DATA.joinpath(f"elec_{idx}.csv"),
            hybrid_assumptions=hybrid_assumptions
        )
        file_path = study_path.joinpath("custom_inputs", f"house_elec_{idx}.txt")
        os.makedirs(file_path.parent, exist_ok=True)
        convert_tsd_to_modelica_txt(
            tsd,
            table_name="Intgainconv_Intgainrad_COPMin",
            columns=["conv", "rad", "COPMin"],
            save_path_file=file_path
        )
        file_path = str(file_path).replace("\\", "//")
        from hps_grid_interaction.bes_simulation.users import get_modifier

        if with_night_set_back:
            dT_set_back = df_users.loc[idx, "dT_set_back"]

        else:
            dT_set_back = 0

        if with_smart_thermostat:
            control_modifier = "redeclare model BuildingSupplySetTemperature = \n" \
                                "    BESMod.Systems.Hydraulical.Control.Components.BuildingSupplyTemperatureSetpoints.SingleZonePID\n" \
                                "      (redeclare HeatPumpSystemGridInteraction.RecordsCollection.PIRoomControlParas parPID)"
        else:
            control_modifier = "redeclare model BuildingSupplySetTemperature = \n" \
                                "    BESMod.Systems.Hydraulical.Control.Components.BuildingSupplyTemperatureSetpoints.IdealHeatingCurve"
        night_set_back_modifier = get_modifier(
            dT_set_back=dT_set_back, night_start=df_users.loc[idx, "night_start"]
        )
        user_modifier = f'userProfiles(fileNameAbsGai=Modelica.Utilities.Files.loadResource("{file_path}"),\n' \
                        f'{night_set_back_modifier}),\n' \
                        f'hydraulic(control({control_modifier}))'

        if with_e_mobility:
            file_path = study_path.joinpath("custom_inputs", f"house_elec_{idx}.txt")
            tsd = TimeSeriesData(
                E_MOBILITY_DATA.joinpath(f"ev_{idx}.csv"),
                sep=","
            )
            tsd.to_float_index()
            tsd.loc[:, "e_mobility"] = tsd.loc[:, ("Ladestrom [kW]", "raw")] * 1000  # to W
            convert_tsd_to_modelica_txt(
                tsd,
                table_name="EMobility",
                columns=["e_mobility"],
                save_path_file=file_path
            )
            file_path = str(file_path).replace("\\", "//")
            e_mobility_modifier = f'fileNameEMob=Modelica.Utilities.Files.loadResource("{file_path}"),' \
                                  f'use_eMob={"true" if with_e_mobility else "false"})'
            user_modifier += "," + e_mobility_modifier
        if non_optimal_heating_curve:
            user_modifier += f",THeaCur_nominal={273.15 + 55}"

        dhw_path = dhw_base_path.joinpath(f"DHWCalc_{idx}.txt")
        if not os.path.exists(dhw_path):
            dhw_path = None
        dhw = {
            "daily_volume": building_data["Anzahl Bewohner"] * dhw_litre_per_person_per_day,
            "time_step": TIME_STEP,
            "path": dhw_path
        }
        year_of_construction = building_data["Baujahr"]
        number_of_floors = building_data["Anzahl Etagen"]
        usage = "single_family_house" if building_data["Gebäudetyp"] == "EFH" else "multi_family_house"
        if usage == "single_family_house":
            net_leased_area = tabula_areas_sfh[year_of_construction]
        else:
            # MFH
            net_leased_area = areas_mfh[year_of_construction] * number_of_floors * flats_per_floor
        if year_of_construction == 2010:
            construction_types = ["tabula_standard"]
        else:
            construction_types = ["tabula_standard", "tabula_retrofit", "tabula_adv_retrofit"]
        for construction_type in construction_types:
            dhw_profiles.append(dhw)
            user_modifiers.append(user_modifier)
            # To create teaser-valid name from "MFH (6 WE)2010" to MFH_6_WE_2010_retrofit
            name = f'{building_data["Gebäudetyp"]}{building_data["Baujahr"]}_' \
                   f'{construction_type.replace("tabula_", "")}'.replace("(", "").replace(")", "_").replace(" ", "_")
            building_configs.append(BuildingConfig(
                name=name,
                year_of_construction=year_of_construction,
                usage=usage,
                number_of_floors=number_of_floors,
                net_leased_area=net_leased_area,
                method="tabula_de",
                with_ahu=False,
                height_of_floors=2.5,
                construction_type=construction_type,
                modify_transfer_system=True,
                number_of_occupants=building_data["Anzahl Bewohner"]
            ))
            excel_for_clustering.append(
                {
                    "Index": idx,
                    "construction_type": construction_type,
                    **building_data,
                }

            )
    pd.DataFrame(excel_for_clustering).set_index("Index").to_excel(
        study_path.joinpath("MonteCarloSimulationInput.xlsx")
    )
    logger.info("Loaded grid and building data")

    return building_configs, user_modifiers, dhw_profiles


def get_bivalence_temperatures(buildings, with_heating_rod: bool, TOda_nominal, model_name: str,
                               hybrid_assumptions=None, cost_optimal_design=False):
    if model_name in ["Monovalent", "HeatDemandCalculation"]:
        if not with_heating_rod:
            return [TOda_nominal] * len(buildings)
        # Bosch and energie-experten recommend TBiv of -6 to -3 for -12 degree TOda_nominal.
        # This value is consistent with own results (see Bergfest and Jahresgespräch).
        return [273.15 + (-6 + (-3)) / 2] * len(buildings)

    T_bivs = []
    T_biv_for_building = {}
    for idx, building in enumerate(buildings):
        # GEG-based design
        if building.name in T_biv_for_building:
            T_bivs.append(T_biv_for_building[building.name])
        else:
            T_bivs.append(274.25)
            T_biv_for_building[building.name] = T_bivs[-1]
    return T_bivs


def get_max_values_in_inputs():
    max_households = []
    for file in os.listdir(HOUSEHOLD_DATA):
        if file.endswith(".csv"):
            df = pd.read_csv(HOUSEHOLD_DATA.joinpath(file), index_col=0)
            max_households.append(df.max())
    max_e_mob = []
    for file in os.listdir(E_MOBILITY_DATA):
        if file.endswith(".csv"):
            df = pd.read_csv(E_MOBILITY_DATA.joinpath(file), index_col=0)
            max_e_mob.append(df.max())
    print("Max household", np.sum(max_households), np.mean(max_households), np.std(max_households))
    print("Max e-mob", np.sum(max_e_mob), np.mean(max_e_mob), np.std(max_e_mob))


def copy_files_for_online_publications(
        destination: Path,
        src_bes: Path = None,
        src_grid: Path = None,
        src_emob: Path = None,
        src_household: Path = None
):
    """
    This function is used to copy relevant result files for the online publication.
    All other plots and intermediate results may be generated with these datapoints.
    If readers require further information, please contact the authors.

    If any of the src folders are None the default ones from __init__.py are used.
    """
    # Building Simulation Results
    extra_case_name_hybrid = "Weather"
    folder_names = []
    from hps_grid_interaction import E_MOBILITY_DATA, HOUSEHOLD_DATA, RESULTS_BES_FOLDER, RESULTS_MONTE_CARLO_FOLDER
    if src_bes is None:
        src_bes = RESULTS_BES_FOLDER
    if src_emob is None:
        src_emob = E_MOBILITY_DATA
    if src_household is None:
        src_household = HOUSEHOLD_DATA
    if src_grid is None:
        src_grid = RESULTS_MONTE_CARLO_FOLDER

    # The same as in __init__.py of repo
    dst_bes = destination.joinpath("01_results", "02_simulations")
    dst_grid = destination.joinpath("01_results", "03_monte_carlo")
    dst_emob = destination.joinpath("time_series_data", "e_mobility")
    dst_household = destination.joinpath("time_series_data", "household")

    # copy_path(src_emob, dst_emob)
    # copy_path(src_household, dst_household)
    #
    # for grid_case in ["oldbuildings", "newbuildings"]:
    #     folder_names.extend([
    #         f"Hybrid{extra_case_name_hybrid}_{grid_case}",
    #         f"Monovalent{extra_case_name_hybrid}_{grid_case}",
    #         f"Monovalent{extra_case_name_hybrid}_{grid_case}_HR",
    #     ])
    # for folder in folder_names:
    #     os.makedirs(dst_bes.joinpath(folder), exist_ok=True)
    #     for name in ["MonteCarloSimulationInputWithEmissions.xlsx", "csv_files"]:
    #         copy_path(src_bes.joinpath(folder, name), dst_bes.joinpath(folder, name))

    # Grid Simulation Results
    from hps_grid_interaction.plotting.plot_loadflow import MONTE_CARLO_METRICS
    files_and_folder_to_copy = {
        "results_to_plot": "results_to_plot",
        "plots": "plots_over_year",
        "plots_detailed_grid_single": "plots_detailed_grid_single",
    }
    for folder in os.listdir(src_grid):
        if (
                folder.startswith("oldbuildings") or folder.startswith("newbuildings")
                and os.path.isdir(src_grid.joinpath(folder))
                and "Analyse" not in folder
        ):
            case_folder = src_grid.joinpath(folder)
            dst_case_folder = dst_grid.joinpath(folder)
            os.makedirs(dst_case_folder, exist_ok=True)
            for file_or_folder in os.listdir(case_folder):
                if file_or_folder in files_and_folder_to_copy or file_or_folder.startswith("monte_carlo_convergence"):
                    copy_path(
                        case_folder.joinpath(file_or_folder),
                        dst_case_folder.joinpath(files_and_folder_to_copy[file_or_folder])
                    )
            for folder_monte_carlo in MONTE_CARLO_METRICS.values():
                monte_carlo_folder_src = case_folder.joinpath(folder_monte_carlo)
                monte_carlo_folder_dst = dst_case_folder.joinpath(folder_monte_carlo)
                os.makedirs(monte_carlo_folder_dst, exist_ok=True)
                for file in os.listdir(monte_carlo_folder_src):
                    if file.startswith("grid_choices") and file.endswith(".png"):
                        copy_path(monte_carlo_folder_src.joinpath(file), monte_carlo_folder_dst.joinpath(file))


def copy_path(src, dst):
    """
    Copy a file or directory from src to dst.

    Parameters:
    src (str): Source path.
    dst (str): Destination path.
    """
    if os.path.isdir(src):
        # It's a directory
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        # It's a file
        shutil.copy(src, dst)
    else:
        raise ValueError("Source path must be a file or directory")


if __name__ == '__main__':
    copy_files_for_online_publications(
        destination=Path(r"E:\fwu\03_paper_reproduction"),
        src_grid=Path(r"X:\Projekte\EBC_ACS0025_EONgGmbH_HybridWP_\Data\04_Ergebnisse\03_monte_carlo"),
    )
