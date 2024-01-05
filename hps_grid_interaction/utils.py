import logging
import os
from typing import Union

from dataclasses import dataclass

import pandas as pd
import numpy as np

from ebcpy import TimeSeriesData
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt
from pathlib import Path


from hps_grid_interaction.bes_simulation.building import BuildingConfig
from hps_grid_interaction.plotting.important_variables import plot_important_variables
from hps_grid_interaction import PROJECT_FOLDER, KERBER_NETZ_XLSX, E_MOBILITY_DATA, HOUSEHOLD_DATA
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
    if assumption == "average":
        return {
            "MFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 75, "tabula_retrofit": 17, "tabula_adv_retrofit": 8},
                1970: {"tabula_standard": 29, "tabula_retrofit": 35, "tabula_adv_retrofit": 36},
                1960: {"tabula_standard": 29, "tabula_retrofit": 35, "tabula_adv_retrofit": 36},
                1950: {"tabula_standard": 29, "tabula_retrofit": 35, "tabula_adv_retrofit": 36},
            },
            "EFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 65, "tabula_retrofit": 29, "tabula_adv_retrofit": 6},
                1970: {"tabula_standard": 35, "tabula_retrofit": 45, "tabula_adv_retrofit": 20},
                1960: {"tabula_standard": 35, "tabula_retrofit": 45, "tabula_adv_retrofit": 20},
                1950: {"tabula_standard": 35, "tabula_retrofit": 45, "tabula_adv_retrofit": 20}
            }
        }
    elif assumption == "no_retrofit":
        return {
            "MFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1970: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1960: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1950: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
            },
            "EFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1970: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1960: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1950: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0}
            }
        }
    elif assumption == "all_adv_retrofit":
        return {
            "MFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1970: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1960: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1950: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
            },
            "EFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1970: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1960: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100},
                1950: {"tabula_standard": 0, "tabula_retrofit": 0, "tabula_adv_retrofit": 100}
            }
        }
    elif assumption == "all_retrofit":
        return {
            "MFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1970: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1960: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1950: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
            },
            "EFH": {
                2010: {"tabula_standard": 100, "tabula_retrofit": 0, "tabula_adv_retrofit": 0},
                1980: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1970: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1960: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0},
                1950: {"tabula_standard": 0, "tabula_retrofit": 100, "tabula_adv_retrofit": 0}
            }
        }
    raise KeyError


def load_outdoor_air_temperature():
    df_oda = pd.read_csv(Path(__file__).parent.joinpath("data", "t_oda.csv"), index_col=0)
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


def load_buildings_and_gains(sheet_name: str, hybrid_assumptions: HybridSystemAssumptions, study_path: Path,
                             with_e_mobility: bool):
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
        night_set_back_modifier = get_modifier(
            dT_set_back=df_users.loc[idx, "dT_set_back"], night_start=df_users.loc[idx, "night_start"]
        )
        user_modifier = f'userProfiles(fileNameAbsGai=Modelica.Utilities.Files.loadResource("{file_path}"),' \
                        f'{night_set_back_modifier})'

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


def extract_electricity_and_save(tsd, path, result_name, with_heating_rod: bool):
    from hps_grid_interaction.bes_simulation.simulation import INIT_PERIOD

    P_heat_pump = "outputs.hydraulic.gen.PEleHeaPum.value"
    P_heating_rod = "outputs.hydraulic.gen.PEleHeaRod.value"
    P_PV = "electrical.generation.internalElectricalPin.PElecGen"
    P_household = "building.internalElectricalPin.PElecLoa"
    P_grid_loa = "electricalGrid.PElecLoa"
    P_grid_gen = "electricalGrid.PElecGen"

    df = tsd.to_df().loc[INIT_PERIOD:] / 1000  # All W to kW, other units will not be selected anyways
    df.index -= df.index[0]
    if len(df.index) != int(365 * 86400 / TIME_STEP + 1):
        logging.error("Not 15 min sampled data: %s", result_name)

    if with_heating_rod:
        df_heat_supply = df.loc[:, P_heat_pump] + df.loc[:, P_heating_rod]
    else:
        df_heat_supply = df.loc[:, P_heat_pump]

    df_to_csv = pd.DataFrame({
        "heat_supply": df_heat_supply,
        "household": df.loc[:, P_household] + df_heat_supply,
        "household+pv": df.loc[:, P_household] + df_heat_supply - df.loc[:, P_PV],
        "household+pv+battery": - df.loc[:, P_grid_loa] - df.loc[:, P_grid_gen],
    })

    os.makedirs(path.joinpath("csv_files"), exist_ok=True)
    df_to_csv.to_csv(path.joinpath("csv_files", result_name.replace(".mat", "_grid_simulation.csv")))


def extract_tsd_results(
        path: Path,
        result_names: list,
        convert_to_hdf_and_delete_mat: bool
):
    logger.debug("Reading file %s", path.name)
    result_names = list(set(result_names))
    try:
        tsd = TimeSeriesData(path)
        result_names = list(set(tsd.get_variable_names()).intersection(result_names))
        tsd = tsd.loc[:, result_names]
    except np.core._exceptions._ArrayMemoryError as err:
        logger.error("Could not read .mat file due to memory-error: %s", err)
        return None  # For DOE, no obj is required.
    logger.debug("Read file %s", path.name)
    if convert_to_hdf_and_delete_mat and path.suffix != ".hdf":
        tsd.save(
            path.parent.joinpath(path.stem + ".hdf"),
            key="DesignOptimization"
        )

    return tsd


def plot_result(tsd, init_period, result_name, save_path, plot_settings):
    plot_important_variables(
        save_path=save_path.joinpath("plots_time", result_name + ".png"),
        x_variable="time",
        scatter=False,
        tsd=tsd,
        init_period=init_period,
        **plot_settings
    )
    plot_important_variables(
        tsd=tsd,
        save_path=save_path.joinpath("plots_scatter", result_name + ".png"),
        x_variable="weaDat.weaBus.TDryBul",
        scatter=True,
        init_period=init_period,
        **plot_settings
    )


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
        if cost_optimal_design:
            THyd_nominal, dTHyd_nominal = building.get_retrofit_temperatures(
                TOda_nominal=TOda_nominal,
                TRoom_nominal=273.15 + 20
            )
            logging.info(f"{THyd_nominal=}, {dTHyd_nominal=}")
            df = TimeSeriesData(
                Path(__file__).parent.joinpath("data", f"GetCOPCurve{int(THyd_nominal)}.mat")
            ).to_df().loc[1:]
            T_bivs.append(
                df.loc[
                    df.loc[:, "sigBusGen.COP"] > hybrid_assumptions.get_minimum_cop(),
                    "TOda"].min()
            )
        else:
            # GEG-based design
            if building.name in T_biv_for_building:
                T_bivs.append(T_biv_for_building[building.name])
            else:
                T_bivs.append(274.25)
                T_biv_for_building[building.name] = T_bivs[-1]
    return T_bivs
