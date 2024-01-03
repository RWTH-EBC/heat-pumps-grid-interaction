import logging
import os
import shutil

from pathlib import Path

import pandas as pd

from hps_grid_interaction.bes_simulation import weather
from hps_grid_interaction.utils import HybridSystemAssumptions
from hps_grid_interaction import utils
from hps_grid_interaction.bes_simulation import simulation
from hps_grid_interaction.bes_simulation.inputs import InputsConfig
from hps_grid_interaction import RESULTS_BES_FOLDER
logger = logging.getLogger(__name__)


def run_simulations(
        model_name: str,
        grid_case: str,
        hybrid_assumptions: HybridSystemAssumptions,
        without_heating_rod: bool = True,
        extract_only: bool = False,
        case_name: str = None,
        n_cpu: int = 8
):
    assert grid_case in ["altbau", "neubau"]
    if case_name is None:
        case_name = model_name
    case_name += f"_{grid_case}"
    if not without_heating_rod:
        case_name += "_HR"
    if extract_only:
        n_cpu = 1

    base_path = RESULTS_BES_FOLDER

    sheet_name = f"Kerber Netz {grid_case.capitalize()}"
    buildings, gains_modifiers, dhw_profiles = utils.load_buildings_and_gains(
        sheet_name=sheet_name,
        study_path=base_path.joinpath(case_name),
        hybrid_assumptions=hybrid_assumptions
    )

    weather_config = weather.WeatherConfig()
    study_path = base_path.joinpath(case_name)

    sim_config = simulation.get_simulation_config(model_name=model_name, without_heating_rod=without_heating_rod)

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
    model_names, result_names = inputs_config.get_model_and_result_names(sim_config.model_name)
    explicit_model_names, new_path = simulation.generate_modelica_package(
        save_path=study_path,
        modifiers=model_names
    )
    sim_config.model_name = explicit_model_names[0]

    sim_api = simulation.start_dymola(
        config=sim_config,
        cd=base_path.joinpath("00_DymolaWorkDir"),
        n_cpu=n_cpu,
        additional_packages=utils.get_additional_packages(inputs_config.buildings) + [new_path],
    )

    if extract_only:
        results = [Path(study_path.joinpath("SimulationResults")).joinpath(result_name + ".mat")
                   for result_name in result_names]
        T_bivs = [None for _ in results]  # Irrelevant here.
    else:
        T_bivs = utils.get_bivalence_temperatures(
            buildings=inputs_config.buildings, model_name=model_name,
            without_heating_rod=without_heating_rod, TOda_nominal=weather_config.TOda_nominal,
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
            savepath="\\\\?\\" + str(sim_api.cd),  # Fix long path issue
            return_option="savepath",
            fail_on_error=False
        )
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
            new_path = study_path.joinpath("SimulationResults", result_name)
            shutil.move(result, new_path)

            result = utils.extract_tsd_results(
                path=Path(new_path),
                result_names=sim_api.result_names,
                convert_to_hdf_and_delete_mat=True
            )
            utils.plot_result(
                tsd=result,
                init_period=simulation.INIT_PERIOD,
                save_path=study_path,
                result_name=result_name.replace(".mat", ""),
                plot_settings=sim_config.plot_settings
            )

            if result is None:
                logging.error("Could not read results, skipping extraction")
                break
            utils.extract_electricity_and_save(tsd=result, path=study_path, result_name=result_name)
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
        return_error = False
    except KeyError as err:
        logging.error(err)
        return_error = True
    finally:
        sim_api.close()
    return return_error


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    HYBRID_ASSUMPTIONS = HybridSystemAssumptions(method="costs")
    KWARGS = dict(
        hybrid_assumptions=HYBRID_ASSUMPTIONS,
        n_cpu=1
    )
    #run_simulations(model_name="Hybrid", case_name="HybridPVBat", grid_case="altbau", **KWARGS)
    run_simulations(model_name="Hybrid", case_name="HybridPVBat", grid_case="neubau", **KWARGS)
    #run_simulations(model_name="Monovalent", case_name="MonovalentPVBat", grid_case="altbau", **KWARGS)
    #run_simulations(model_name="Monovalent", case_name="MonovalentPVBat", grid_case="neubau", **KWARGS)
    #run_simulations(model_name="Monovalent", case_name="MonovalentPVBat", grid_case="altbau", without_heating_rod=True, **KWARGS)
    #run_simulations(model_name="Monovalent", case_name="MonovalentPVBat", grid_case="neubau", without_heating_rod=True, **KWARGS)
