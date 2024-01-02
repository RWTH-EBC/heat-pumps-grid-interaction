within HeatPumpSystemGridInteraction.HybridHeatPumpSystem;
model HeatDemandCalculation
  "Model to calculate the heat demand"
  extends BESMod.Systems.BaseClasses.PartialBuildingEnergySystem(
    redeclare BESMod.Systems.Electrical.ElectricalSystem electrical(
      redeclare BESMod.Systems.Electrical.Generation.NoGeneration generation,
      redeclare BESMod.Systems.Electrical.Distribution.DirectlyToGrid
        distribution,
      redeclare BESMod.Systems.Electrical.Transfer.IdealHeater transfer,
      redeclare BESMod.Systems.Electrical.Control.IdealHeater control),
    redeclare HybridHeatPumpSystem.BaseClasses.CustomTEASERThermalZone building,
    redeclare BESMod.Systems.Control.NoControl control,
    redeclare BESMod.Systems.Hydraulical.HydraulicSystem hydraulic(
      redeclare final BESMod.Systems.Hydraulical.Generation.NoGeneration
        generation,
      redeclare BESMod.Systems.Hydraulical.Control.NoControl control,
      redeclare BESMod.Systems.Hydraulical.Distribution.BuildingOnly distribution(
          nParallelDem=1),
      redeclare final BESMod.Systems.Hydraulical.Transfer.NoHeatTransfer transfer(
          nParallelSup=1)),
    redeclare BESMod.Systems.Demand.DHW.StandardProfiles DHW(
      redeclare BESMod.Systems.Demand.DHW.RecordsCollection.ProfileM DHWProfile,
      redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum,
      redeclare BESMod.Systems.Demand.DHW.TappingProfiles.calcmFlowEquStatic
        calcmFlow),
    redeclare
      .HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.CustomInputs
      userProfiles,
    redeclare HeatPumpSystemGridInteraction.RecordsCollection.SystemParameters systemParameters(
      QBui_flow_nominal=building.QRec_flow_nominal,
      use_hydraulic=false,
      use_elecHeating=true),
    redeclare
      HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.ParameterStudy
      parameterStudy,
    redeclare final package MediumDHW = AixLib.Media.Water,
    redeclare final package MediumZone = AixLib.Media.Air,
    redeclare final package MediumHyd = AixLib.Media.Water,
    redeclare BESMod.Systems.Ventilation.NoVentilation ventilation);

  extends Modelica.Icons.Example;
  Modelica.Blocks.Sources.RealExpression reaExpCOP[systemParameters.nZones](y=electrical.transfer.heaKPI.u ./ (0.5 *heatingCurve.TSet/max(heatingCurve.TSet - heatingCurve.TOda, 0.2)))
    annotation (Placement(transformation(extent={{-400,0},{-380,20}})));
  BESMod.Systems.Hydraulical.Control.Components.HeatingCurve heatingCurve(
    TSup_nominal=systemParameters.THydSup_nominal[1],
    TRet_nominal=systemParameters.THydSup_nominal[1] - 10,
    TOda_nominal=systemParameters.TOda_nominal,
    nHeaTra=1.3,
    TZoneSet=systemParameters.TSetZone_nominal[1])
    annotation (Placement(transformation(extent={{-380,40},{-360,60}})));
  AixLib.BoundaryConditions.WeatherData.Bus weaBus1
    annotation (Placement(transformation(extent={{-424,84},{-404,104}})));
  BESMod.Utilities.KPIs.EnergyKPICalculator elKPI[systemParameters.nZones]
    annotation (Placement(transformation(extent={{-340,0},{-320,20}})));
equation
  connect(electrical.weaBus, weaBus1) annotation (Line(
      points={{-198,103.429},{-306,103.429},{-306,94},{-414,94}},
      color={255,204,51},
      thickness=0.5));
  connect(heatingCurve.TOda, weaBus1.TDryBul) annotation (Line(points={{-382,50},
          {-398,50},{-398,48},{-410,48},{-410,94},{-414,94}}, color={0,0,127}),
      Text(
      string="%second",
      index=1,
      extent={{-6,3},{-6,3}},
      horizontalAlignment=TextAlignment.Right));
  connect(elKPI.u, reaExpCOP.y)
    annotation (Line(points={{-341.8,10},{-379,10}}, color={0,0,127}));
  connect(elKPI.KPI, outputs.PHeaEl) annotation (Line(points={{-317.8,10},{-310,
          10},{-310,-134},{285,-134},{285,0}}, color={135,135,135}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  annotation (experiment(
      StopTime=31536000,
      Interval=600,
      __Dymola_Algorithm="Dassl"));
end HeatDemandCalculation;
