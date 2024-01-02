import os
import itertools
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ebcpy.preprocessing import convert_datetime_index_to_float_index
from ebcpy import TimeSeriesData

from hps_grid_interaction.boundary_conditions import weather
from hps_grid_interaction.plotting.config import PlotConfig

from utils import HybridSystemAssumptions


COLUMNS_EMISSIONS = [
    'constant_gas', 'constant_electricity', '2025_Dynamic_gas',
    '2025_Dynamic_electricity', '2025_Static_gas',
    '2025_Static_electricity', '2030_Dynamic_gas',
    '2030_Dynamic_electricity', '2030_Static_gas',
    '2030_Static_electricity', '2037_Dynamic_gas',
    '2037_Dynamic_electricity', '2037_Static_gas',
    '2037_Static_electricity'
]


def load_data(interpolate: bool = False):
    df = pd.read_excel(Path(__file__).parent.joinpath("data", "Results_price_emissions_updated.xlsx"),
                       sheet_name="All",
                       index_col=0,
                       header=[0, 1, 2])
    if not interpolate:
        df.index *= 3600
        return df
    # Convert the seconds-based index to a Datetime index
    df['DateTime'] = pd.to_datetime(df.index, unit='h')
    df.set_index('DateTime', inplace=True)

    # Resample to 15-minute intervals and interpolate
    df_resampled = df.resample('15T').asfreq().interpolate(method='linear')
    col = "Average Emissions [g_CO2/kWh]"
    years = {}
    df = convert_datetime_index_to_float_index(df_resampled)
    for year in [2025, 2030, 2037]:
        average = df.loc[:, (year, col)].columns[0]
        _df = df.loc[:, (year, col, average)]
        years[year] = {"dynamic": _df, "static": average}
    return years, df.index[-1]


def calc_all_emissions():
    hybrid_assumptions = {
        "constant": HybridSystemAssumptions(method="costs"),
        **{str(year): HybridSystemAssumptions(method="costs", emissions_electricity=str(year))
           for year in [2025, 2030, 2037]}
    }
    calc_emissions("HybridGEGBiv_altbau", hybrid_assumptions, file_ending=".hdf")
    calc_emissions("HybridGEGBiv_neubau", hybrid_assumptions, file_ending=".hdf")
    # calc_emissions("Monovalent_altbau_HR", hybrid_assumptions, file_ending=".hdf")
    # calc_emissions("Monovalent_altbau", hybrid_assumptions, file_ending=".hdf")
    # calc_emissions("Hybrid_altbau", hybrid_assumptions, file_ending=".hdf")
    # calc_emissions("Hybrid_COP3_neubau", hybrid_assumptions, file_ending=".mat")
    # calc_emissions("Hybrid_COP2_neubau", hybrid_assumptions, file_ending=".hdf")
    # calc_emissions("Monovalent_neubau", hybrid_assumptions, file_ending=".mat")
    # calc_emissions("Hybrid_neubau", hybrid_assumptions, file_ending=".mat")


def calc_emissions(case: str, hybrid_assumptions: Dict[str, HybridSystemAssumptions], file_ending=".hdf"):
    print(f"Extracting case {case}")
    gas_name = "outputs.hydraulic.dis.PBoiAftBuf.value"
    p_el_hr_name = "outputs.hydraulic.gen.PEleHeaRod.value"
    p_el_hp_name = "outputs.hydraulic.gen.PEleHeaPum.value"
    COP_name = "hydraulic.generation.sigBusGen.COP"
    Q_boi_name = "outputs.hydraulic.dis.QBoi_flow.integral"

    path = Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations").joinpath(case)
    df_sim = pd.read_excel(
        path.joinpath("MonteCarloSimulationInput.xlsx"),
        sheet_name="Sheet1"
    )

    path_sim = path.joinpath("SimulationResults")
    hybrid = "Hybrid" in path.name
    col = "Average Emissions [g_CO2/kWh]"
    years, _until = load_data(interpolate=True)
    for file in os.listdir(path_sim):
        if not file.endswith(file_ending):
            continue
        tsd = TimeSeriesData(path_sim.joinpath(file)).to_df()
        tsd = tsd.loc[86400 * 2:]
        tsd.index -= 86400 * 2
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
            Q_renewable = np.sum(Q_hea_pum + tsd.loc[:, p_el_hr_name] * 0.97) * 0.25
        else:
            tsd_electricity = tsd.loc[:, p_el_hp_name] / 1000
            Q_renewable = np.sum(Q_hea_pum) * 0.25
        if hybrid:
            Q_boi = tsd.iloc[-1][Q_boi_name] / 3600
            percent_renewables = Q_renewable / (Q_renewable + Q_boi) * 100
            df_sim.loc[idx, "QBoi"] = Q_boi
        else:
            percent_renewables = 100
        df_sim.loc[idx, "QRenewable"] = Q_renewable
        df_sim.loc[idx, "percent_renewables"] = percent_renewables

        def populate_data_to_dict(_df, _idx, assumption, case, gas, elec):
            _df.loc[_idx, f"{assumption}{case}_gas"] = gas
            _df.loc[_idx, f"{assumption}{case}_electricity"] = elec
            return _df

        for assumption_name, hybrid_assumption in hybrid_assumptions.items():
            e_gas = tsd_gas.sum() * hybrid_assumption.emissions_natural_gas * 0.25 / 1000  # in kg
            if isinstance(hybrid_assumption.emissions_electricity, str):
                # Dynamic emissions
                emission_values = years[int(hybrid_assumption.emissions_electricity)]["dynamic"]
                e_electricity = (
                        np.sum(np.multiply(tsd_electricity, emission_values.values)) * 0.25  # in g
                        / 1000  # in kg
                )
                #plt.figure()
                ###plt.scatter(emission_values.values, tsd_electricity * 0.25)
                #plt.title(assumption_name)
                #plt.ylabel("$P_\mathrm{el}$ in kWh")
                #plt.xlabel("$e_\mathrm{el}$ in g/kWh")
                populate_data_to_dict(df_sim, idx, assumption_name, "_Dynamic", e_gas, e_electricity)
                # Static emissions
                emission_value = years[int(hybrid_assumption.emissions_electricity)]["static"]
                e_electricity = np.sum(tsd_electricity) * 0.25 * emission_value / 1000  # in kg
                populate_data_to_dict(df_sim, idx, assumption_name, "_Static", e_gas, e_electricity)
            else:
                e_electricity = np.sum(tsd_electricity) * 0.25 * hybrid_assumption.emissions_electricity / 1000  # in kg
                populate_data_to_dict(df_sim, idx, assumption_name, "", e_gas, e_electricity)
            #plt.show()
            #raise Exception
    df_sim = df_sim.set_index("Index")
    df_sim.to_excel(
        path.joinpath("MonteCarloSimulationInputWithEmissions.xlsx"),
        sheet_name="Sheet1"
    )


def plot_emissions_scatter():
    df_emissions = load_data()
    weather_config = weather.WeatherConfig()
    df_weather = convert_datetime_index_to_float_index(weather_config.get_hourly_weather_data())
    col = "Average Emissions [g_CO2/kWh]"
    df_weather = df_weather.loc[:df_emissions.index[-1]]
    df_weather["renewables"] = (
            df_weather["WindSpeed"] / df_weather["WindSpeed"].max() +
            df_weather["DirNormRad"] / df_weather["DirNormRad"].max()
    )

    labels = {
        "DryBulbTemp": "$T_\mathrm{Oda}$ in °C",
        "DirNormRad": "$H_\mathrm{Dir}$ in W/m2K",
        "WindSpeed": "Wind speed in in m/s",
        "renewables": "Wind and solar radiation in -",
    }
    plt.scatter(df_weather["DryBulbTemp"], df_weather["WindSpeed"])
    plt.figure()
    plt.scatter(df_weather["DryBulbTemp"], df_weather["DirNormRad"])

    for variable in [2025, 2030, 2037]:
        for sensor in ["DryBulbTemp", "WindSpeed", "DirNormRad", "renewables"]:
            fig, ax = plt.subplots(1, 1, sharey=True)
            ax.set_title(f"Jahr: {variable} | Wetterdaten: {sensor}")
            ax.set_xlabel(labels[sensor])
            ax.set_ylabel(f"Emissions in g/kWh")
            ax.scatter(df_weather.loc[:, sensor], df_emissions.loc[:, (variable, col)])
            fig.tight_layout()
            fig.savefig(Path(__file__).parent.joinpath("plots", f"Emissions_{sensor}_{variable}.png"))

    plt.show()


def get_all_cases_iteration_product():
    tech_cases = ["Hybrid", "Monovalent"]
    grid_cases = ["altbau", "neubau"]
    quota_cases = [
        "average", "no_retrofit",
        "all_retrofit", "all_adv_retrofit"
    ]
    iterations = []
    for tech, grid_case, quota in itertools.product(tech_cases, grid_cases, quota_cases):
        if grid_case == "altbau" and tech == "Monovalent":
            hr = [True, False]
        else:
            hr = [False]
        for with_hr in hr:
            iterations.append((tech, grid_case, quota, with_hr))
    return iterations


def get_case_name(grid_case: str, quota: str, with_hr: bool, tech: str = None):
    hr = "_HR" if with_hr else ""
    case = grid_case + hr + "_" + quota
    if tech is None:
        return case
    else:
        return tech + "_" + case


def get_emission_options():
    options = []
    for key in COLUMNS_EMISSIONS:
        options.append(key.replace("_gas", "").replace("_electricity", ""))
    return list(set(options))


def aggregate_and_save_all_cases(
        lastfluss_xlsx: Path,
        emissions_json: Path,
        skip_emissions: bool = False
):
    res = {}
    df_lastfluss = pd.read_excel(lastfluss_xlsx, index_col=0)
    with open(emissions_json, "r") as file:
        results = json.load(file)
    # Initialize Result DataFrame
    # Create a MultiIndex
    index = pd.MultiIndex.from_tuples([
        ('Inputs', 'Technologie'),
        ('Inputs', 'Baualter'),
        ('Inputs', 'Sarnierungsquote'),
        ('Inputs', 'Heizstab'),
        ('Inputs', 'Transformator'),
        #('Netzbelastung', 'min_V_pu'),
        #('Emissionen', '2022_Static'),
        #('GEG', 'percent_renewables'),
    ], names=['', 'Index'])

    def get_label_and_factor(metric):
        if metric == "max":
            return "Maximaler Strompeak Wärmeerzeugung in kW", 1
        elif metric == "sum":
            return "Strombedarf Wärmeerzeugung in MWh", 1e-3
        raise ValueError

    from monte_carlo import get_short_case_name
    # Create a DataFrame with the MultiIndex
    df = pd.DataFrame(columns=index)
    trafos = [400, 630, 1000, 2000]
    idx = -1
    for tech, grid_case, quota, with_hr in get_all_cases_iteration_product():
        for trafo in trafos:
            case_name = get_case_name(grid_case, quota, with_hr)
            column_lastfluss = get_short_case_name(case_name=case_name, tech=tech) + f"_{trafo}"
            if column_lastfluss not in df_lastfluss.columns:
                continue
            idx += 1
            df.loc[idx, ('Inputs', 'Technologie')] = tech
            df.loc[idx, ('Inputs', 'Baualter')] = grid_case
            df.loc[idx, ('Inputs', 'Sarnierungsquote')] = quota
            df.loc[idx, ('Inputs', 'Heizstab')] = with_hr
            df.loc[idx, ('Inputs', 'Transformator')] = trafo
            metrics_lastfluss = {
                "min_V_pu": "Minimale Spannung in p.u.",
                "max_line": "Maximale Belastung Leitung in %",
                "time_smaller_95": "Anteil Spannung unter 0.95 p.u. in %",
                "time_smaller_97": "Anteil Spannung unter 0.97 p.u. in %",
            }
            for df_lastfluss_idx, label in metrics_lastfluss.items():
                df.loc[idx, ("Netzbelastung", label)] = df_lastfluss.loc[df_lastfluss_idx, column_lastfluss]
            for metric in ["max", "sum"]:
                label, factor = get_label_and_factor(metric)
                value = results[case_name]["grid"][tech][metric]["ONT"] * factor
                df.loc[idx, ("Netzbelastung", label)] = value
            # Build sum of gas and electricity
            df.loc[idx, ("GEG", f"Wärme aus Gas in MWh")] = results[case_name]["emissions"][tech]["QBoi"] / 1e6
            df.loc[idx, ("GEG", f"Wärme aus Strom in MWh")] = results[case_name]["emissions"][tech]["QRenewable"] / 1e6
            if skip_emissions:
                continue
            for option in get_emission_options():
                option_name = " ".join(option.split("_"))
                option_name = option_name.replace("constant", "2022 Static")
                df.loc[idx, ("Emissionen", f"{option_name} Gas in tCO2")] = results[case_name]["emissions"][tech][option + "_gas"] / 1000
                df.loc[idx, ("Emissionen", f"{option_name} Strom in tCO2")] = results[case_name]["emissions"][tech][option + "_electricity"] / 1000
                df.loc[idx, ("Emissionen", f"{option_name} in tCO2")] = (
                        df.loc[idx, ("Emissionen", option_name + " Strom in tCO2")] +
                        df.loc[idx, ("Emissionen", option_name + " Gas in tCO2")]
                )

    # Re-calculate, sum makes no sense
    df.loc[:, ("GEG", "Anteil Erneuerbar in %")] = (
            df.loc[:, ("GEG", "Wärme aus Strom in MWh")] /
            (df.loc[:, ("GEG", "Wärme aus Strom in MWh")] + df.loc[:, ("GEG", "Wärme aus Gas in MWh")])
    ) * 100
    # Save:
    df.to_excel(emissions_json.parent.joinpath("AggregatedResults.xlsx"))



if __name__ == '__main__':
    PlotConfig.load_default()  # Trigger rc_params
    #calc_all_emissions()
    aggregate_and_save_all_cases(
        lastfluss_xlsx=Path(r"D:\01_Projekte\09_HybridWP\01_Results\03_lastfluss\LastflussSimulationenGEGBiv-RONT\3-ph\analysis.xlsx"),
        emissions_json=Path(r"D:\01_Projekte\09_HybridWP\01_Results\02_simulations\results_to_plot.json"),
        skip_emissions=True
    )
