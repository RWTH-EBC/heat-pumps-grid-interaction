within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model GetCOPCurve
  extends BESMod.Utilities.HeatGeneration.PartialGetHeatGenerationCurve(
    redeclare BESMod.Examples.UseCaseDesignOptimization.AachenSystem
      systemParameters(THydSup_nominal={THyd_nominal}),
    redeclare
      HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.HeatPumpAndElectricHeater
      generation(
      redeclare BESMod.Systems.Hydraulical.Generation.RecordsCollection.DefaultHP
        parHeaPum,
      redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum,
      redeclare BESMod.Systems.RecordsCollection.TemperatureSensors.DefaultSensor
        parTemSen,
      use_eleHea=false,
      redeclare BESMod.Systems.Hydraulical.Generation.RecordsCollection.DefaultHR
        parEleHea,
      redeclare HeatPumpSystemGridInteraction.RecordsCollection.VitoCal250
        dataTable),
    ramp(
      height=34,
      duration=84400,
      offset=273.15 - 15,
      startTime=1000),
    realExpression(y=generation.heatPump.con.QFlow_in),
    heatingCurve(TRet_nominal=THyd_nominal - dTHyd_nominal));
  parameter Modelica.Units.SI.Temperature THyd_nominal=273.15+55 "Nominal radiator temperature";
  parameter Modelica.Units.SI.TemperatureDifference dTHyd_nominal=10 "Nominal radiator temperature difference";

  annotation (experiment(StopTime=86400, __Dymola_Algorithm="Dassl"));
end GetCOPCurve;
