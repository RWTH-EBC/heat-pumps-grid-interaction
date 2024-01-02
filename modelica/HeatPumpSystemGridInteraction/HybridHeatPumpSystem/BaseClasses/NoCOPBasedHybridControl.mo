within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model NoCOPBasedHybridControl "Decides when to use the boiler"
  extends PartialCOPBasedHybridControl;

equation
  connect(priGenOnSet, priGenOn)
    annotation (Line(points={{-120,80},{110,80}}, color={255,0,255}));
  connect(secGenOn, secGenOnSet) annotation (Line(points={{110,0},{-94,0},{-94,46},
          {-120,46}}, color={255,0,255}));
  annotation (Diagram(coordinateSystem(extent={{-100,-100},{100,100}})), Icon(
        coordinateSystem(extent={{-100,-100},{80,100}})));
end NoCOPBasedHybridControl;
