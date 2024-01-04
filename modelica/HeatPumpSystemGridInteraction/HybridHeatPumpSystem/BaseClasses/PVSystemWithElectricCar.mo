within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model PVSystemWithElectricCar
  extends BESMod.Systems.Electrical.Generation.PVSystemMultiSub(realToElecCon(
        use_souLoa=use_eMob));
  Modelica.Blocks.Sources.CombiTimeTable tabEMobility(
    final tableOnFile=true,
    final extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
    final tableName="EMobility",
    final fileName=fileNameEMob,
    columns={2}) if use_eMob
    "Profiles for internal gains of machines and lights in W" annotation (
      Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=180,
        origin={-50,70})));
  parameter Boolean use_eMob=true "= true to activate e mobility"
    annotation (Dialog(tab="E-Mobility"));
  parameter String fileNameEMob=Modelica.Utilities.Files.loadResource(
      "modelica://HeatPumpSystemGridInteraction/HybridHeatPumpSystem/CustomInputs.txt")
                                "File where data for e mobility is stored"
    annotation (Dialog(enable=use_eMob, tab="E-Mobility"));
equation
  connect(tabEMobility.y[1], realToElecCon.PEleLoa)
    annotation (Line(points={{-39,70},{0,70},{0,46},{46,46}}, color={0,0,127}));
end PVSystemWithElectricCar;
