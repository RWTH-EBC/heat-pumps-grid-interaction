within HeatPumpSystemGridInteraction;
package BaseClasses
  extends Modelica.Icons.BasesPackage;

  partial model PartialSystem2D "Partial system with 2D table data"
    extends BESMod.Systems.BaseClasses.PartialBuildingEnergySystem(
    redeclare BESMod.Systems.Demand.Building.TEASERThermalZone building(
        ABui=sum(building.zoneParam.AZone),
        hBui=sum(building.zoneParam.VAir) ./ sum(building.zoneParam.AZone),
        ARoo=sum(building.zoneParam.ARoof),
        redeclare HeatPumpSystemGridInteraction.RecordsCollection.Retrofit1918_SingleDwelling
          oneZoneParam,
        final zoneParam=fill(building.oneZoneParam, building.nZones),
        final ventRate=fill(0, building.nZones),
        final use_verboseEnergyBalance=true),
      final use_openModelica=false,
      redeclare HeatPumpSystemGridInteraction.RecordsCollection.SystemParameters systemParameters(
          QBui_flow_nominal=building.QRec_flow_nominal, THydSup_nominal={
            THyd_nominal}),
      redeclare final package MediumDHW = AixLib.Media.Water,
      redeclare final package MediumZone = AixLib.Media.Air,
      redeclare final package MediumHyd = AixLib.Media.Water,
      weaDat(final computeWetBulbTemperature=true),
      redeclare BESMod.Systems.Ventilation.NoVentilation ventilation);
    parameter Modelica.Units.SI.Temperature THyd_nominal=273.15+55 "Nominal radiator temperature";
    parameter Modelica.Units.SI.TemperatureDifference dTHyd_nominal=10 "Nominal radiator temperature difference";
    parameter Real scalingFactor;

    parameter Modelica.Units.SI.HeatFlowRate QHeaPumNomCosFun_flow = Modelica.Blocks.Tables.Internal.getTable2DValueNoDer2(
          tabQConFlo,
          35,
          2) * scalingFactor
          "QHeaPum_flow at A2W35, a typical point of manufacturer data. 80 % compressor speed is 
        used, as manufacturers typically sell devices at part load for higher COP.";

    parameter Modelica.Units.SI.HeatFlowRate QPriAtTOdaNom_flow_nominal=Modelica.Blocks.Tables.Internal.getTable2DValueNoDer2(
          tabQConFlo,
          THyd_nominal - 273.15,
          systemParameters.TOda_nominal - 273.15) * scalingFactor annotation(Evaluate=true);
    parameter Modelica.Units.SI.HeatFlowRate QHeaPumBiv_flow=Modelica.Blocks.Tables.Internal.getTable2DValueNoDer2(
          tabQConFlo,
          THydAtBiv_nominal - 273.15,
          parameterStudy.TBiv - 273.15) annotation(Evaluate=true);

    parameter Modelica.Units.SI.Temperature THydAtBiv_nominal=
      THyd_nominal +
      (- 1 / (systemParameters.TSetZone_nominal[1] - systemParameters.TOda_nominal) *
      ((2 * THyd_nominal - dTHyd_nominal)/2 - systemParameters.TSetZone_nominal[1]) *
      1 / hydraulic.transfer.nHeaTra + dTHyd_nominal / 2 *
      (- 1 / (systemParameters.TSetZone_nominal[1] - systemParameters.TOda_nominal))) *
      (parameterStudy.TBiv - systemParameters.TOda_nominal);
    parameter Modelica.Blocks.Types.ExternalCombiTable2D tabQConFlo=
        Modelica.Blocks.Types.ExternalCombiTable2D(
        "NoName",
        "NoName",
        hydraulic.generation.heatPump.innerCycle.PerformanceDataHPHeating.dataTable.tableQdot_con,
        hydraulic.generation.heatPump.innerCycle.PerformanceDataHPHeating.smoothness,
        Modelica.Blocks.Types.Extrapolation.LastTwoPoints,
        false) "External table object";
    parameter Modelica.Blocks.Types.ExternalCombiTable2D tabPEle=
        Modelica.Blocks.Types.ExternalCombiTable2D(
        "NoName",
        "NoName",
        hydraulic.generation.heatPump.innerCycle.PerformanceDataHPHeating.dataTable.tableP_ele,
        hydraulic.generation.heatPump.innerCycle.PerformanceDataHPHeating.smoothness,
        Modelica.Blocks.Types.Extrapolation.LastTwoPoints,
        false) "External table object";

    annotation (experiment(
        StopTime=31536000,
        Interval=600,
        __Dymola_Algorithm="Dassl"));
  end PartialSystem2D;

end BaseClasses;
