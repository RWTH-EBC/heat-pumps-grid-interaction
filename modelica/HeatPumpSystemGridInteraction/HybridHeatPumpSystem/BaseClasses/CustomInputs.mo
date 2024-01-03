within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model CustomInputs "Custom inputs for project"
  extends BESMod.Systems.UserProfiles.BaseClasses.PartialUserProfiles;
  parameter String fileNameIntGains=Modelica.Utilities.Files.loadResource("modelica://BESMod/Resources/InternalGains.txt")
    "File where matrix is stored";
  parameter String fileNameAbsGai=Modelica.Utilities.Files.loadResource("modelica://HeatPumpSystemGridInteraction/HybridHeatPumpSystem/CustomInputs.txt")
    "File where matrix is stored";
  parameter Boolean use_nigSetBac=false;
  parameter Real houNigEnd(unit="h")=6 "Start hour of night set-back"
    annotation (Dialog(enable=use_nigSetBac));
  parameter Real houNigStart(unit="h")=22 "End hour of night set-back"
    annotation (Dialog(enable=use_nigSetBac));
  parameter Modelica.Units.SI.TemperatureDifference dTNigSetBac=2
    "Temperature delta for night set-back"
    annotation (Dialog(enable=use_nigSetBac));
  Modelica.Blocks.Sources.CombiTimeTable tabIntGai(
    final tableOnFile=true,
    final extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
    final tableName="Internals",
    final fileName=fileNameIntGains,
    columns=2:4) "Profiles for internal gains" annotation (Placement(
        transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={-10,30})));

  Modelica.Blocks.Math.Gain gainIntGai[3](k={1,0,0})
                                                  "Gain for internal gains"
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={30,30})));

  Modelica.Blocks.Sources.Constant conTSetZone[nZones](k=TSetZone_nominal)
    if not use_nigSetBac "Constant room set temperature"
                                    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={-10,-50})));
  Modelica.Blocks.Sources.CombiTimeTable tabGaiLigMach(
    final tableOnFile=true,
    final extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
    final tableName="Intgainconv_Intgainrad_COPMin",
    final fileName=fileNameAbsGai,
    columns={2,3,4}) "Profiles for internal gains of machines and lights in W"
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={-10,-10})));
  Modelica.Blocks.Sources.Pulse nigSetBakTSetZone[nZones](
    amplitude=2,
    width=100*(houNigStart - houNigEnd)/24,
    period=3600*24,
    offset=TSetZone_nominal .- dTNigSetBac,
    startTime=3600*houNigEnd) if use_nigSetBac "Constant room set temperature"
    annotation (Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={-10,-90})));
equation
  connect(tabIntGai.y, gainIntGai.u)
    annotation (Line(points={{1,30},{18,30}}, color={0,0,127}));
  connect(gainIntGai.y, useProBus.intGains) annotation (Line(points={{41,30},{
          74,30},{74,-1},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  connect(conTSetZone.y, useProBus.TZoneSet) annotation (Line(points={{1,-50},{
          115,-50},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  connect(tabGaiLigMach.y[1], useProBus.absIntGaiConv) annotation (Line(points={{1,
          -10},{74,-10},{74,-1},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  connect(tabGaiLigMach.y[2], useProBus.absIntGaiRad) annotation (Line(points={{1,-10},
          {74,-10},{74,-1},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  connect(tabGaiLigMach.y[3], useProBus.COPMin) annotation (Line(points={{1,-10},{
          74,-10},{74,-1},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
  connect(nigSetBakTSetZone.y, useProBus.TZoneSet) annotation (Line(points={{1,-90},
          {30,-90},{30,-92},{115,-92},{115,-1}}, color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{6,3},{6,3}},
      horizontalAlignment=TextAlignment.Left));
end CustomInputs;
