import pathlib
from pydantic import BaseModel, FilePath
from aixweather.imports.TRY import load_try_from_file

from hps_grid_interaction import DATA_PATH


class WeatherConfig(BaseModel):
    dat_file: FilePath = DATA_PATH.joinpath("TRY2015_523845130645_Jahr.dat")
    TOda_nominal: float = 273.15 - 12.6
    mos_path: FilePath = DATA_PATH.joinpath("TRY2015_523845130645_Jahr.mos")

    def get_modelica_modifier(self):
        mos_file = str(self.mos_path).replace("\\", "//")
        return f'systemParameters(\n' \
               f'    filNamWea=Modelica.Utilities.Files.loadResource("{mos_file}"),\n' \
               f'    TOda_nominal={self.TOda_nominal})'

    def get_name(self, location_name=False):
        _path = pathlib.Path(self.dat_file)
        if location_name:
            return str(_path.parents[1].stem).replace(" ", "_") + _path.stem.split("_")[-1]
        return pathlib.Path(self.dat_file).stem

    def get_hourly_weather_data(self):
        return load_try_from_file(path=str(self.dat_file))
