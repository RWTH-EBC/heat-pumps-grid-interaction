import logging
from typing import List
from pathlib import Path

from teaser.project import Project

from pydantic import BaseModel, FilePath, Field
import numpy as np

logger = logging.getLogger(__name__)


def create_buildings(
        name: str,
        buildings: List["BuildingConfig"],
        export: bool = True,
        path: Path = None
):
    """
    Generate the buildings using teaser and possibly export it

    :param str name:
        Name of the TEASER project
    :param List["BuildingConfig"] buildings:
        List of buildings to create
    :param bool export:
        Whether to export the modelica files or not
    :param Path path:
        Path to save the export

    :return: List["BuildingConfig"]
        Modified list of buildings with calculated outputs
    """

    prj = Project(load_data=True)

    prj.name = name
    buildings_to_export = {}
    for building_config in buildings:
        if building_config.name not in buildings_to_export:
            buildings_to_export[building_config.name] = building_config

    logger.debug("Creating %s buildings, %s are duplicates",
                 len(buildings), len(buildings) - len(buildings_to_export))

    for building_config in buildings_to_export.values():
        prj.add_residential(
            name=building_config.name,
            method=building_config.method,
            usage=building_config.usage,
            construction_type=building_config.construction_type,
            number_of_floors=building_config.number_of_floors,
            height_of_floors=building_config.height_of_floors,
            with_ahu=building_config.with_ahu,
            year_of_construction=building_config.year_of_construction,
            net_leased_area=building_config.net_leased_area,
        )

    # export building model (see Teaser/project)
    prj.used_library_calc = 'AixLib'
    prj.number_of_elements_calc = 4  # Default value
    prj.calc_all_buildings(raise_errors=True)
    for bui in prj.buildings:
        for zone in bui.thermal_zones:
            zone.use_conditions.with_heating = False
            zone.use_conditions.with_cooling = False

    for teaser_bui, building_config in zip(prj.buildings, buildings_to_export.values()):
        building_config.heat_load_outside_factor = teaser_bui.thermal_zones[0].model_attr.heat_load_outside_factor
        building_config.heat_load_ground_factor = teaser_bui.thermal_zones[0].model_attr.heat_load_ground_factor
        if building_config.number_of_occupants is not None:
            teaser_bui.thermal_zones[0].use_conditions.persons = (
                building_config.number_of_occupants / teaser_bui.thermal_zones[0].area
            )

    for building_config in buildings:
        building_config.heat_load_outside_factor = buildings_to_export[building_config.name].heat_load_outside_factor
        building_config.heat_load_ground_factor = buildings_to_export[building_config.name].heat_load_ground_factor

    if export:
        if path is None:
            raise TypeError("For export, you need to specify a path")
        package_path = prj.export_aixlib(path=path.joinpath("Buildings"))
        for building_config in buildings:
            building_config.record_name = ".".join([
                name,
                f"{building_config.name}",
                f"{building_config.name}_DataBase",
                f"{building_config.name}_SingleDwelling"
            ])
            building_config.package_path = Path(package_path).joinpath("package.mo")
    return buildings


class BuildingConfig(BaseModel):
    """
    See TEASER documentation for more information on parameters.
    Fields with default values are calculated in this tool using TEASER:
    """

    name: str
    method: str
    usage: str
    construction_type: str
    number_of_floors: int
    height_of_floors: float
    with_ahu: bool
    year_of_construction: int
    net_leased_area: float
    record_name: str = None
    package_path: FilePath = None
    modify_transfer_system: bool = False
    number_of_occupants: int = None
    heat_load_outside_factor: float = Field(
        title="UA value to ambient air, calculated by TEASER",
        default=None
    )
    heat_load_ground_factor: float = Field(
        title="UA value to soil, calculated by TEASER",
        default=None
    )

    def get_modelica_modifier(self, input_config: "InputConfig"):
        THyd_nominal, dTHyd_nominal = self.get_retrofit_temperatures(
            TOda_nominal=input_config.weather.TOda_nominal,
            TRoom_nominal=input_config.user.room_set_temperature
        )

        TNonRetrofit_nominal, _, _ = self.get_nominal_supply_temperature(self.year_of_construction)

        if TNonRetrofit_nominal == 35:
            transfer_system_type = \
                "BESMod.Systems.Hydraulical.Transfer.UFHTransferSystem transfer(" \
                "    nHeaTra=1.3, " \
                "    redeclare BESMod.Systems.Hydraulical.Transfer.RecordsCollection.SteelRadiatorStandardPressureLossData parTra, " \
                "    redeclare BESMod.Systems.Hydraulical.Transfer.RecordsCollection.DefaultUFHData UFHParameters, " \
                "    redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum)"
        else:
            transfer_system_type =\
                "BESMod.Systems.Hydraulical.Transfer.IdealValveRadiator transfer(" \
                "    redeclare BESMod.Systems.Hydraulical.Transfer.RecordsCollection.SteelRadiatorStandardPressureLossData parTra, " \
                "    redeclare BESMod.Systems.Hydraulical.Transfer.RecordsCollection.RadiatorTransferData parRad, " \
                "    redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum)"

        if self.modify_transfer_system:
            modifier = f"  hydraulic(redeclare {transfer_system_type}),\n"
        else:
            modifier = ""
        return f"{modifier}building(redeclare {self.record_name} oneZoneParam),\n" \
               f"  THyd_nominal={THyd_nominal},\n" \
               f"  dTHyd_nominal={dTHyd_nominal}"

    def get_name(self):
        return self.record_name.split(".")[-1]

    def get_statistical_parameters(self) -> dict:
        return {
            "area": self.net_leased_area,
            "year": self.year_of_construction,
            "retrofit": int(self.construction_type == "tabula_retrofit")
        }

    def get_heating_load(
            self,
            TOda_nominal: float,
            TRoom_nominal: float = 293.15,
            TSoil: float = 286.15
    ):
        """
        From TEASER:
        heat_load_outside_factor : float [W/K]
            Factor needed for recalculation of the heat load of the thermal zone.
            This can be used to recalculate the thermalzones heat load inside
            Modelica export for parametric studies. This works only together with
            heat_load_ground_factor.

            heat_load = heat_load_outside_factor * (t_inside - t_outside) +
            heat_load_ground_factor * (t_inside - t_ground).
        heat_load_ground_factor : float [W/K]
            Factor needed for recalculation of the heat load of the thermal zone.
            This can be used to recalculate the thermalzones heat load inside
            Modelica export for parametric studies. See heat_load_outside_factor.
        """
        if self.heat_load_ground_factor is None:
            raise ValueError("You first have to create the building models using"
                             "`create_buildings` to calculate the heat load")
        return (
                self.heat_load_outside_factor * (TRoom_nominal - TOda_nominal) +
                self.heat_load_ground_factor * (TRoom_nominal - TSoil)
        )

    @staticmethod
    def get_nominal_supply_temperature(year_of_construction):
        """
        Source:
        https://www.ffe.de/projekte/waermepumpen-fahrplan-finanzielle-kipppunkte-zur-modernisierung-mit-waermepumpen-im-wohngebaeudebestand/
        """
        # TODO: Single function
        if year_of_construction < 1950:  # TODO: Possible 1960 is better.
            return 90 + 273.15, 20, 1.3  # TODO: Check values
        if year_of_construction < 1990:
            return 70 + 273.15, 15, 1.3  # TODO: Check values
        if year_of_construction < 2010:
            return 55 + 273.15, 10, 1.3  # TODO: Check values
        return 35 + 273.15, 5, 1  # As in BESMod

    def get_retrofit_temperatures(
            self,
            TOda_nominal: float,
            TRoom_nominal: float
    ):
        """
        According to Lämmle et al. 2022, Chapter 4.1
        """
        # TODO: Single function?
        t_supply, dT_supply, n_heat_exponent = self.get_nominal_supply_temperature(
            year_of_construction=self.year_of_construction
        )
        if self.construction_type == "tabula_standard":
            return t_supply, dT_supply
        self_without_retrofit = self.copy()
        self_without_retrofit.construction_type = "tabula_standard"
        self_without_retrofit = create_buildings(
            name="Temporary",
            buildings=[self_without_retrofit],
            export=False
        )[0]
        QNom2 = self.get_heating_load(TOda_nominal=TOda_nominal, TRoom_nominal=TRoom_nominal)
        QNom1 = self_without_retrofit.get_heating_load(TOda_nominal=TOda_nominal, TRoom_nominal=TRoom_nominal)
        return self._get_new_supply_temperature(
            TRoom1=TRoom_nominal,
            TRoom2=TRoom_nominal,
            n=n_heat_exponent,
            TSup1=t_supply,
            TRet1=t_supply - dT_supply,
            QNom1=QNom1,
            QNom2=QNom2
        )

    @staticmethod
    def _get_new_supply_temperature(
            TRoom1: float,
            TRoom2: float,
            n: float,
            TSup1: float,
            TRet1: float,
            QNom1: float,
            QNom2: float
    ):
        """
        This helper function uses the exact formulation from
        "Lämmle et al. 2022, Chapter 4.1" to avoid confusion.

        :param TRoom1:
            Old room temperature
        :param TRoom2:
            New room temperature
        :param n:
            Heat transfer exponent
        :param TSup1:
            Old supply temperature
        :param TRet1:
            Old return temperature
        :param QNom1:
            Old heating load
        :param QNom2:
            New heating load

        :return: T2:
            New supply temperature
        :return: T2:
            New return temperature
        """
        # TODO: Single function
        dT2 = (TSup1 - TRet1) * QNom2 / QNom1
        dTLog2 = (
                (TSup1 - TRet1) /
                np.log((TSup1 - TRoom1) / (TRet1 - TRoom1)) *
                (QNom2 / QNom1) ** (1 / n)
        )
        TSup2 = TRoom2 + dT2 * (np.exp(dT2 / dTLog2) / (np.exp(dT2 / dTLog2) - 1))
        return max(308.15, TSup2), max(5.0, dT2)
