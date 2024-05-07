import os
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from ebcpy.preprocessing import convert_datetime_index_to_float_index
from ebcpy import TimeSeriesData

from hps_grid_interaction import DATA_PATH
from hps_grid_interaction.utils import HybridSystemAssumptions
from hps_grid_interaction.bes_simulation.simulation import INIT_PERIOD


logger = logging.getLogger(__name__)

gas_name = "outputs.hydraulic.dis.PBoiAftBuf.value"
p_el_hr_name = "outputs.hydraulic.gen.PEleEleHea.value"
p_el_hp_name = "outputs.hydraulic.gen.PEleHeaPum.value"
p_el_hr_int_name = "outputs.hydraulic.gen.PEleEleHea.integral"
p_el_hp_int_name = "outputs.hydraulic.gen.PEleHeaPum.integral"
COP_name = "hydraulic.generation.sigBusGen.COP"
Q_boi_name = "outputs.hydraulic.dis.QBoi_flow.integral"
ufh_name = "outputs.hydraulic.tra.QUFH_flow[1].integral"
A_name = "building.zoneParam[1].AZone"
heat_load_name = "systemParameters.QBui_flow_nominal[1]"
building_demand_name = "outputs.building.QTraGain[1].integral"
dhw_demand_name = "outputs.DHW.Q_flow.integral"
TOda_nom_name = "systemParameters.TOda_nominal"
hea_rod_nom_name = "hydraulic.generation.eleHea.Q_flow_nominal"
THyd_name = "THyd_nominal"
hea_rod_eta_name = "hydraulic.generation.parEleHea.eta"
P_PV = "electrical.generation.internalElectricalPin.PElecGen"
P_household = "building.internalElectricalPin.PElecLoa"
P_grid_loa = "electricalGrid.PElecLoa"
P_grid_gen = "electricalGrid.PElecGen"


VARIABLE_NAMES = [
    gas_name,
    p_el_hr_name,
    p_el_hp_name,
    p_el_hr_int_name,
    p_el_hp_int_name,
    COP_name,
    Q_boi_name,
    ufh_name,
    A_name,
    heat_load_name,
    building_demand_name,
    dhw_demand_name,
    TOda_nom_name,
    hea_rod_nom_name,
    THyd_name,
    hea_rod_eta_name,
    P_PV,
    P_household,
    P_grid_loa,
    P_grid_gen,
]

COLUMNS_EMISSIONS = [
    'constant_gas', 'constant_electricity', '2025_Dynamic_gas',
    '2025_Dynamic_electricity', '2025_Static_gas',
    '2025_Static_electricity', '2030_Dynamic_gas',
    '2030_Dynamic_electricity', '2030_Static_gas',
    '2030_Static_electricity', '2037_Dynamic_gas',
    '2037_Dynamic_electricity', '2037_Static_gas',
    '2037_Static_electricity'
]


def extract_electricity_and_save(tsd, path, result_name, with_heating_rod: bool):
    df = tsd.to_df().loc[INIT_PERIOD:] / 1000  # All W to kW, other units will not be selected anyways
    df.index -= df.index[0]
    if len(df.index) != int(365 * 86400 / TIME_STEP + 1):
        logging.error("Not 15 min sampled data: %s", result_name)

    if with_heating_rod:
        df_heat_supply = df.loc[:, p_el_hp_name] + df.loc[:, p_el_hr_name]
    else:
        df_heat_supply = df.loc[:, p_el_hp_name]

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
        try:
            tsd = TimeSeriesData(path, variable_names=result_names)
        except KeyError as key_err:
            logger.info("Could not find variables in .mat file: %s", key_err)
            from ebcpy.modelica.simres import loadsim
            available_variables = loadsim(path)
            result_names = list(set(available_variables.keys()).intersection(result_names))
            tsd = TimeSeriesData(path, variable_names=result_names)
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


def load_emission_data(interpolate: bool = False):
    df = pd.read_excel(DATA_PATH.joinpath("Results_price_emissions_updated.xlsx"),
                       sheet_name="All",
                       index_col=0,
                       header=[0, 1, 2])
    if not interpolate:
        df.index *= 3600
        return df
    # Convert the seconds-based index to a Datetime index
    df['DateTime'] = pd.to_datetime(df.index, unit='h')
    df.set_index('DateTime', inplace=True)
    from hps_grid_interaction.bes_simulation.simulation import TIME_STEP
    interval = int(TIME_STEP / 60)
    # Resample to minute intervals and interpolate
    df_resampled = df.resample(f'{interval}min').asfreq().interpolate(method='linear')
    col = "Average Emissions [g_CO2/kWh]"
    years = {}
    df = convert_datetime_index_to_float_index(df_resampled)
    for year in [2025, 2030, 2037]:
        average = df.loc[:, (year, col)].columns[0]
        _df = df.loc[:, (year, col, average)]
        years[year] = {"dynamic": _df, "static": average}
    return years, df.index[-1]


def calc_emissions(case: str, hybrid_assumptions: Dict[str, HybridSystemAssumptions], file_ending=".hdf"):
    print(f"Extracting case {case}")

    from hps_grid_interaction import RESULTS_BES_FOLDER
    from hps_grid_interaction.bes_simulation.simulation import INIT_PERIOD, W_to_Wh

    path = RESULTS_BES_FOLDER.joinpath(case)
    df_sim = pd.read_excel(
        path.joinpath("MonteCarloSimulationInput.xlsx"),
        sheet_name="Sheet1"
    )

    path_sim = path.joinpath("SimulationResults")
    hybrid = "Hybrid" in path.name
    col = "Average Emissions [g_CO2/kWh]"
    years, _until = load_emission_data(interpolate=True)
    for file in os.listdir(path_sim):
        if not file.endswith(file_ending):
            continue
        tsd = TimeSeriesData(path_sim.joinpath(file)).to_df()
        tsd = tsd.loc[INIT_PERIOD:]
        tsd.index -= INIT_PERIOD
        tsd = tsd.loc[:_until]
        idx = int(file.split("_")[0])
        with_hr = p_el_hr_name in tsd
        if hybrid:
            tsd_gas = tsd.loc[:, gas_name] / 1000
        else:
            tsd_gas = np.zeros(8760 * 4)
        Q_hea_pum = tsd.loc[:, p_el_hp_name] * tsd.loc[:, COP_name]
        if with_hr:
            tsd_electricity = tsd.loc[:, p_el_hp_name] / 1000 + tsd.loc[:, p_el_hr_name] / 1000
            Q_renewable = np.sum(Q_hea_pum + tsd.loc[:, p_el_hr_name] * 0.97) * W_to_Wh
        else:
            tsd_electricity = tsd.loc[:, p_el_hp_name] / 1000
            Q_renewable = np.sum(Q_hea_pum) * W_to_Wh
        if hybrid:
            Q_boi = tsd.iloc[-1][Q_boi_name] / 3600
            percent_renewables = Q_renewable / (Q_renewable + Q_boi) * 100
            df_sim.loc[idx, "QBoi"] = Q_boi
        else:
            percent_renewables = 100
        df_sim.loc[idx, "QRenewable"] = Q_renewable
        df_sim.loc[idx, "percent_renewables"] = percent_renewables

        # Analysis of heat demand and nominal heat load for plausibility analysis
        if file_ending != ".mat":
            raise FileNotFoundError(".mat files needed for detailed building plausibility study.")
        tsd_loc = tsd.iloc[-1]  # Last row for integral and parameters
        heat_load = tsd_loc[heat_load_name]
        building_demand = tsd_loc[building_demand_name]
        dhw_demand = tsd_loc[dhw_demand_name]
        if ufh_name in tsd.columns:
            # ufh loss is negative, thus, subtract
            building_demand -= tsd_loc[ufh_name]
        df_sim.loc[idx, "building_demand"] = building_demand / 3600000
        heat_demand = building_demand + dhw_demand
        df_sim.loc[idx, "heat_demand"] = heat_demand / 3600000
        df_sim.loc[idx, "dhw_demand"] = dhw_demand / 3600000
        df_sim.loc[idx, "ABui"] = tsd_loc[A_name]
        df_sim.loc[idx, "heat_load"] = heat_load
        if with_hr:
            W_el_ges = tsd_loc[p_el_hr_int_name] + tsd_loc[p_el_hp_int_name]
        else:
            W_el_ges = tsd_loc[p_el_hp_int_name]
        df_sim.loc[idx, "SCOP_Sys"] = heat_demand / W_el_ges
        df_sim.loc[idx, "WEleGen"] = W_el_ges / 3600000
        THyd_nominal = tsd_loc[THyd_name]
        TOda_nominal = tsd_loc[TOda_nom_name]
        if with_hr:
            PEleHeaMax = tsd_loc[hea_rod_nom_name] / tsd_loc[hea_rod_eta_name]
        else:
            PEleHeaMax = 0
        PEleHeaPumMax = tsd_loc["scalingFactor"] * 3398
        df_sim.loc[idx, "PEleMax"] = PEleHeaPumMax + PEleHeaMax

        def populate_data_to_dict(_df, _idx, assumption, case, gas, elec):
            _df.loc[_idx, f"{assumption}{case}_gas"] = gas
            _df.loc[_idx, f"{assumption}{case}_electricity"] = elec
            return _df

        for assumption_name, hybrid_assumption in hybrid_assumptions.items():
            e_gas = tsd_gas.sum() * hybrid_assumption.emissions_natural_gas * W_to_Wh / 1000  # in kg
            if isinstance(hybrid_assumption.emissions_electricity, str):
                # Dynamic emissions
                emission_values = years[int(hybrid_assumption.emissions_electricity)]["dynamic"]
                e_electricity = (
                        np.sum(np.multiply(tsd_electricity, emission_values.values)) * W_to_Wh  # in g
                        / 1000  # in kg
                )
                #plt.figure()
                ###plt.scatter(emission_values.values, tsd_electricity * W_to_Wh)
                #plt.title(assumption_name)
                #plt.ylabel("$P_\mathrm{el}$ in kWh")
                #plt.xlabel("$e_\mathrm{el}$ in g/kWh")
                populate_data_to_dict(df_sim, idx, assumption_name, "_Dynamic", e_gas, e_electricity)
                # Static emissions
                emission_value = years[int(hybrid_assumption.emissions_electricity)]["static"]
                e_electricity = np.sum(tsd_electricity) * W_to_Wh * emission_value / 1000  # in kg
                populate_data_to_dict(df_sim, idx, assumption_name, "_Static", e_gas, e_electricity)
            else:
                e_electricity = np.sum(tsd_electricity) * W_to_Wh * hybrid_assumption.emissions_electricity / 1000  # in kg
                populate_data_to_dict(df_sim, idx, assumption_name, "", e_gas, e_electricity)
            #plt.show()
            #raise Exception
    df_sim = df_sim.set_index("Index")
    df_sim.to_excel(
        path.joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
        sheet_name="Sheet1"
    )
