from pathlib import Path

from ebcpy import DymolaAPI
from typing import List
from pydantic import BaseModel, FilePath
from hps_grid_interaction import BESMOD_PATH, MODELICA_PATH


INIT_PERIOD = 86400 * 2
TIME_STEP = 900
# Used to convert results in W into Wh.
# As a results in one time-step, e.g. x W holds for this time-step,
# the integral is x * TIME_STEP Ws. Converted to Wh gives x / 3600 Wh.
W_to_Wh = TIME_STEP / 3600


class SimulationConfig(BaseModel):
    model: str
    sim_setup: dict
    packages: List[FilePath] = []
    result_names: list = []
    plot_settings: dict = {}
    convert_to_hdf_and_delete_mat: bool = True


def generate_modelica_package(save_path: Path, modifiers: list):
    package_content = f'''package ModelsToSimulate\n'''
    explicit_model_names = []
    for i, modifier in enumerate(modifiers, start=1):
        package_content += f'  model Case{i}\n' \
                           f'    extends {modifier};\n' \
                           f'  end Case{i};\n'
        explicit_model_names.append(f"ModelsToSimulate.Case{i}")
    package_content += 'end ModelsToSimulate;\n'
    new_path = save_path.joinpath('ModelsToSimulate.mo')
    with open(save_path.joinpath('ModelsToSimulate.mo'), 'w') as f:
        f.write(package_content)
    return explicit_model_names, new_path


def generate_mos_script(config: SimulationConfig, additional_packages: list, save_path_mos: Path):
    with open(BESMOD_PATH, "r") as file:
        lines = file.readlines()
    lines.append("\n")
    for package in config.packages + additional_packages:
        clean_path = str(package).replace("\\", "//")
        lines.append(f'openModel("{clean_path}", changeDirectory=false);\n')
    with open(save_path_mos, "w") as file:
        file.writelines(lines)


def get_simulation_config(model, with_heating_rod):
    import json
    with open("plots/hybrid_plot_config.json", "r") as file:
        plot_config = json.load(file)

    y_variables = {
        "$T_\mathrm{Oda}$ in °C": "weaDat.weaBus.TDryBul",
        "$T_\mathrm{Room}$ in °C": ["hydraulic.buiMeaBus.TZoneMea[1]", "hydraulic.useProBus.TZoneSet[1]"],
        "$y_\mathrm{Val}$ in %": "hydraulic.transfer.outBusTra.opening[1]",
        "$T_\mathrm{DHW}$ in °C": ["hydraulic.distribution.sigBusDistr.TStoDHWBotMea",
                                   "hydraulic.distribution.sigBusDistr.TStoDHWTopMea"],
        "$T_\mathrm{Buf}$ in °C": ["hydraulic.distribution.sigBusDistr.TStoBufBotMea",
                                   "hydraulic.distribution.sigBusDistr.TStoBufTopMea"],
        "$T_\mathrm{HeaPum}$ in °C": ["hydraulic.generation.sigBusGen.THeaPumIn",
                                      "hydraulic.generation.sigBusGen.THeaPumOut"],
        "$COP$ in -": "hydraulic.generation.sigBusGen.COP",
        "$y_\mathrm{HeaPum}$ in %": "hydraulic.generation.sigBusGen.yHeaPumSet",
        "$\dot{Q}_\mathrm{DHW}$ in kW": "outputs.DHW.Q_flow.value",
        "$\dot{Q}_\mathrm{Bui}$ in kW": "outputs.building.QTraGain[1].value",
        "$P_\mathrm{el,HeaPum}$": "outputs.hydraulic.gen.PEleHeaPum.value",
    }
    if model == "Hybrid":
        y_variables.update({
            "$y_\mathrm{Boi}$ in %": "hydraulic.distribution.sigBusDistr.yBoi",
            "$T_\mathrm{BoiOut}$ in °C": "hydraulic.distribution.sigBusDistr.TBoiOut",
            "$\dot{Q}_\mathrm{Boi}$": "outputs.hydraulic.dis.QBoi_flow.value",
        })
    elif with_heating_rod:
        y_variables.update({"$P_\mathrm{el,EleHea}$": "outputs.hydraulic.gen.PEleEleHea.value"})

    plot_settings = dict(
        x_vertical_lines=["parameterStudy.TBiv"],
        plot_config=plot_config,
        y_variables=y_variables
    )

    return SimulationConfig(
        model=f"HeatPumpSystemGridInteraction.HybridHeatPumpSystem.{model}",
        sim_setup=dict(stop_time=86400 * 365, output_interval=TIME_STEP),
        result_names=[],
        packages=[MODELICA_PATH],
        convert_to_hdf_and_delete_mat=True,
        plot_settings=plot_settings
    )


def start_dymola(
        config: SimulationConfig,
        cd: Path,
        n_cpu,
        additional_packages: list = None,
        save_path_mos: Path = None
):
    if additional_packages is None:
        additional_packages = []
    packages = config.packages + additional_packages

    if save_path_mos is not None:
        generate_mos_script(
            config=config,
            additional_packages=additional_packages,
            save_path_mos=save_path_mos
        )

    dym_api = DymolaAPI(
        cd=cd,
        model_name=config.model,
        mos_script_pre=BESMOD_PATH,
        packages=list(set(packages)),
        n_cpu=n_cpu,
        show_window=True,
        debug=False,
        modify_structural_parameters=False
    )
    dym_api.model_name = config.model
    dym_api.set_sim_setup(config.sim_setup)
    dym_api.sim_setup.stop_time += INIT_PERIOD
    from hps_grid_interaction.plotting.important_variables import get_names_of_plot_variables

    result_names_to_plot = get_names_of_plot_variables(
        x_variable=config.plot_settings.get("x_variable", ""),
        y_variables=config.plot_settings.get("y_variables", {}),
        x_vertical_lines=config.plot_settings.get("x_vertical_lines", [])
    )

    result_names = list(dym_api.outputs.keys())
    result_names.extend(config.result_names)
    result_names.extend(result_names_to_plot)

    dym_api.result_names = list(set(result_names))

    return dym_api
