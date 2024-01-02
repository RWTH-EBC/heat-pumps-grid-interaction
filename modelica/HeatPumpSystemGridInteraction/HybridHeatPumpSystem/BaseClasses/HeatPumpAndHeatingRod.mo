within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model HeatPumpAndHeatingRod "Monovalent heat pump with COP measurement"
  extends BESMod.Systems.Hydraulical.Generation.HeatPumpAndHeatingRod(
    final TSoilConst=283.15,
    redeclare final package Medium_eva = IBPSA.Media.Air,
    final use_airSource=true,
    redeclare model PerDataMainHP =
        AixLib.DataBase.HeatPump.PerformanceData.LookUpTable2D (dataTable=
            dataTable, extrapolation=false));
  Modelica.Blocks.Sources.RealExpression reaExpCOP(y=heatPump.innerCycle.PerformanceDataHPHeating.Qdot_ConTable.y
        /heatPump.innerCycle.PerformanceDataHPHeating.P_eleTable.y)
    "COP measurement"
    annotation (Placement(transformation(extent={{20,20},{40,40}})));
  replaceable parameter AixLib.DataBase.HeatPump.HeatPumpBaseDataDefinition
    dataTable "Data Table of HP" annotation(choicesAllMatching=true);
equation
  connect(reaExpCOP.y, sigBusGen.COP) annotation (Line(points={{41,30},{46,30},{
          46,16},{2,16},{2,98}},
                              color={0,0,127}), Text(
      string="%second",
      index=1,
      extent={{-3,6},{-3,6}},
      horizontalAlignment=TextAlignment.Right));
end HeatPumpAndHeatingRod;
