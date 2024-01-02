within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
partial model PartialCOPBasedHybridControl "Decides when to use the boiler"

  parameter Modelica.Units.SI.Temperature TBiv "Bivalence temperature";
  parameter Modelica.Units.SI.Temperature TCutOff "Cutoff temperature";

  Modelica.Blocks.Interfaces.RealInput COP "Current COP"
    annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
  Modelica.Blocks.Interfaces.BooleanOutput secGenOn
    "Turn secondary generator on"
    annotation (Placement(transformation(extent={{100,-10},{120,10}})));
  Modelica.Blocks.Interfaces.RealInput COPMin "Minimal COP to run heat pump"
    annotation (Placement(transformation(extent={{-140,-60},{-100,-20}})));
  Modelica.Blocks.Interfaces.BooleanInput priGenOnSet
    "Primary generator on signal"
    annotation (Placement(transformation(extent={{-140,60},{-100,100}})));
  Modelica.Blocks.Interfaces.BooleanInput secGenOnSet
    "Secondary generator on signal"
    annotation (Placement(transformation(extent={{-140,26},{-100,66}})));
  Modelica.Blocks.Interfaces.BooleanOutput priGenOn
    "Turn secondary generator on"
    annotation (Placement(transformation(extent={{100,70},{120,90}})));
  annotation (Diagram(coordinateSystem(extent={{-100,-100},{100,100}})), Icon(
        coordinateSystem(extent={{-100,-100},{80,100}})));
end PartialCOPBasedHybridControl;
