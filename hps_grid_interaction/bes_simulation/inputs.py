from pydantic import BaseModel
from typing import List

from hps_grid_interaction.bes_simulation import weather
from hps_grid_interaction.bes_simulation.dhw import DHWCalcConfig
from hps_grid_interaction.bes_simulation.building import BuildingConfig


class InputsConfig(BaseModel):
    buildings: List[BuildingConfig]
    dhw_profiles: List[DHWCalcConfig]
    modifiers: List[str]
    weather: weather.WeatherConfig

    def get_model_and_result_names(self, model_name):
        result_names = []
        model_names = []
        for idx, building in enumerate(self.buildings):
            result_names.append(f"{idx}_{building.get_name()}")
            model_names.append(model_name + self.__get_modelica_modifier(idx=idx))

        return model_names, result_names

    def __get_modelica_modifier(self, idx: int):
        modifiers = [
            self.buildings[idx].get_modelica_modifier(TOda_nominal=self.weather.TOda_nominal),
            self.dhw_profiles[idx].get_modelica_modifier(),
            self.weather.get_modelica_modifier(),
            self.modifiers[idx]
        ]
        merged_modifier = ",\n  ".join(
            [modifier for modifier in modifiers if modifier]
        )
        return f"({merged_modifier})"
