import logging
import os

from pathlib import Path

import pandas as pd

from hps_grid_interaction.bes_simulation import weather
from hps_grid_interaction.utils import HybridSystemAssumptions
from hps_grid_interaction import utils
from hps_grid_interaction.bes_simulation import simulation
from hps_grid_interaction.bes_simulation.inputs import InputsConfig
from hps_grid_interaction import RESULTS_BES_FOLDER
from hps_grid_interaction.bes_simulation import result_processing
from hps_grid_interaction.plotting.important_variables import plot_result

logger = logging.getLogger(__name__)


def run_simulations(
        model_name: str,
        grid_case: str,
        hybrid_assumptions: HybridSystemAssumptions,
        with_heating_rod: bool = False,
        with_e_mobility: bool = False,
        with_night_set_back: bool = False,
        with_smart_thermostat: bool = True,
        non_optimal_heating_curve: bool = False,
        extract_only: bool = False,
        case_name: str = None,
        n_cpu: int = 8
):
    assert grid_case in ["oldbuildings", "newbuildings"]
    if case_name is None:
        case_name = model_name
    case_name += f"_{grid_case}"
    if with_heating_rod:
        case_name += "_HR"
    if extract_only:
        n_cpu = 1

    base_path = RESULTS_BES_FOLDER

    sheet_name = f"Kerber Netz {grid_case}"
    buildings, gains_modifiers, dhw_profiles = utils.load_buildings_and_gains(
        sheet_name=sheet_name,
        study_path=base_path.joinpath(case_name),
        hybrid_assumptions=hybrid_assumptions,
        with_e_mobility=with_e_mobility,
        non_optimal_heating_curve=non_optimal_heating_curve,
        with_night_set_back=with_night_set_back,
        with_smart_thermostat=with_smart_thermostat
    )

    weather_config = weather.WeatherConfig()
    study_path = base_path.joinpath(case_name)

    sim_config = simulation.get_simulation_config(model=model_name, with_heating_rod=with_heating_rod)

    inputs_config = InputsConfig(
        weather=weather_config,
        modifiers=gains_modifiers,
        buildings=buildings,
        dhw_profiles=dhw_profiles,
    )
    from hps_grid_interaction.bes_simulation.building import create_buildings
    inputs_config.buildings = create_buildings(
        path=study_path,
        name="Buildings_" + case_name,
        buildings=inputs_config.buildings
    )
    model_names, result_names = inputs_config.get_model_and_result_names(sim_config.model)
    explicit_model_names, new_path = simulation.generate_modelica_package(
        save_path=study_path,
        modifiers=model_names
    )
    sim_config.model = explicit_model_names[0]

    sim_api = simulation.start_dymola(
        config=sim_config,
        working_directory=base_path.joinpath("00_DymolaWorkDir"),
        n_cpu=n_cpu,
        additional_packages=utils.get_additional_packages(inputs_config.buildings) + [new_path],
        save_path_mos=study_path.joinpath("open_models.mos")
    )

    sim_results_to_extract = sim_api.result_names + [
        "outputs.hydraulic.gen.PEleHeaPum.value",
        "outputs.hydraulic.gen.PEleEleHea.value",
        "electrical.generation.internalElectricalPin.PElecGen",
        "building.internalElectricalPin.PElecLoa",
        "electricalGrid.PElecLoa",
        "electricalGrid.PElecGen",
    ] + result_processing.VARIABLE_NAMES
    sim_results_to_extract = list(set(sim_results_to_extract))

    if extract_only:
        results = [Path(study_path.joinpath("SimulationResults")).joinpath(result_name + ".mat")
                   for result_name in result_names]
        T_bivs = [None for _ in results]  # Irrelevant here.
    else:
        T_bivs = utils.get_bivalence_temperatures(
            buildings=inputs_config.buildings, model_name=model_name,
            with_heating_rod=with_heating_rod, TOda_nominal=weather_config.TOda_nominal,
            hybrid_assumptions=hybrid_assumptions, cost_optimal_design=False
        )
        pd.DataFrame(dict(
            model_names=model_names,
            result_names=result_names,
            T_bivs=T_bivs
        )).to_excel(study_path.joinpath("simulation_inputs.xlsx"))
        results = sim_api.simulate(
            parameters=[{"parameterStudy.TBiv": t_biv} for t_biv in T_bivs],
            result_file_name=result_names,
            model_names=explicit_model_names,
            savepath="\\\\?\\" + str(study_path.joinpath("SimulationResults")),  # Fix long path issue
            return_option="savepath",
            fail_on_error=False
        )
    sim_api.close()
    try:
        os.makedirs(study_path.joinpath("SimulationResults"), exist_ok=True)
        results_last_points = []
        failed_data = []
        for result, model_name, t_biv in zip(results, model_names, T_bivs):
            if result is None or not Path(result).exists():
                failed_data.append(dict(
                    model_name=model_name,
                    T_biv=t_biv
                ))
                continue
            result_name = Path(result).name

            result = result_processing.extract_tsd_results(
                path=Path(result),
                result_names=sim_results_to_extract,
                convert_to_hdf_and_delete_mat=True
            )
            plot_result(
                tsd=result,
                init_period=simulation.INIT_PERIOD,
                save_path=study_path,
                result_name=result_name.replace(".mat", ""),
                plot_settings=sim_config.plot_settings
            )

            if result is None:
                logging.error("Could not read results, skipping extraction")
                break
            result_processing.extract_electricity_and_save(
                tsd=result, path=study_path, result_name=result_name,
                with_heating_rod=with_heating_rod
            )
            df = result.to_df()

            # Extract only last points:
            res = {}
            for result_name in df.columns:
                if (
                        result_name.startswith("outputs.building.dTComHea[") or
                        result_name.startswith("outputs.building.dTComCoo[") or
                        result_name.startswith("outputs.hydraulic.ctrl.dTComHea_adjust") or
                        result_name.startswith("outputs.hydraulic.ctrl.dTComHea_abs") or
                        result_name.endswith(".integral")
                ):
                    res[result_name] = df.iloc[-1][result_name] - df.loc[simulation.INIT_PERIOD, result_name]
                else:
                    res[result_name] = df.iloc[-1][result_name]
            results_last_points.append(res)
        pd.DataFrame(results_last_points).to_excel(study_path.joinpath("Results.xlsx"))
        if failed_data:
            pd.DataFrame(failed_data).to_excel(study_path.joinpath("failed_simulations.xlsx"))
    except (KeyError, AttributeError) as err:
        logging.error(err)   
    extract_monte_carlo_xlsx(case_name)


def extract_monte_carlo_xlsx(case_name):
    # Calc emissions
    hybrid_assumptions = {
        "constant": HybridSystemAssumptions(method="costs"),
        **{str(year): HybridSystemAssumptions(method="costs", emissions_electricity=str(year))
           for year in [2025, 2030, 2037]}
    }
    result_processing.postprocessing(case_name, hybrid_assumptions, file_ending=".hdf")


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    HYBRID_ASSUMPTIONS = HybridSystemAssumptions(method="costs")
    KWARGS = dict(
        hybrid_assumptions=HYBRID_ASSUMPTIONS,
        n_cpu=25,
        extract_only=True,
        with_smart_thermostat=False,
        non_optimal_heating_curve=True
    )
    for GRID in [
        #"newbuildings",
        "oldbuildings"
    ]:
        # run_simulations(model_name="Hybrid", case_name="HybridHC", grid_case=GRID, **KWARGS)
        # run_simulations(model_name="Monovalent", case_name="MonovalentHC", grid_case=GRID, with_heating_rod=True, **KWARGS)
        # run_simulations(model_name="Monovalent", case_name="MonovalentHC", grid_case=GRID, with_heating_rod=False, **KWARGS)
        extract_monte_carlo_xlsx(case_name=f"MonovalentHC_{GRID}_HR")
        # extract_monte_carlo_xlsx(case_name=f"MonovalentHC_{GRID}")
        # extract_monte_carlo_xlsx(case_name=f"HybridHC_{GRID}")
