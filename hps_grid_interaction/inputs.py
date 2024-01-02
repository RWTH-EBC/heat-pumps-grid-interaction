from pydantic import BaseModel
from typing import List

from hps_grid_interaction.boundary_conditions import weather
from hps_grid_interaction.boundary_conditions.dhw import DHWCalcConfig


class InputsConfig(BaseModel):
    buildings: dict
    dhw_profiles: List[DHWCalcConfig]
    modifiers: dict
    weather: weather.WeatherConfig
    users: dict

    def get_model_and_result_names(self, model_name):
        result_names = []
        model_names = []
        for idx, building in enumerate(self.buildings):
            result_names.append(f"{idx}_{building.get_name()}_{self.users[idx].get_name()}")
            model_names.append(model_name + self.__get_modelica_modifier(idx=idx))

        return result_names, model_names

    def __get_modelica_modifier(self, idx: int):
        modifiers = [
            self.buildings[idx].get_modelica_modifier(input_config=self),
            self.dhw_profiles[idx].get_modelica_modifier(),
            self.weather.get_modelica_modifier(),
            self.users[idx].get_modelica_modifier(),
        ]
        if self.modifier:
            modifiers.append(self.modifier)
        merged_modifier = ",\n  ".join(
            [modifier for modifier in modifiers if modifier]
        )
        return f"({merged_modifier})"
