within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
model COPBasedHybridControl "Decides when to use the boiler"
  extends PartialCOPBasedHybridControl;

  Modelica.Blocks.Logical.Hysteresis lesCOPMin(
    final uLow=0,
    final uHigh=0.05,
    final pre_y_start=false) "If COP is lower than COPMin, turn boiler on"
    annotation (Placement(transformation(extent={{-40,-30},{-20,-10}})));
  Modelica.Blocks.Math.Add sub(final k1=+1, final k2=-1)
    "Subtract COP and COPMin"
    annotation (Placement(transformation(extent={{-82,-30},{-62,-10}})));
  Modelica.Blocks.Logical.Not not1 "If COP is lower than COPMin, turn boiler on"
    annotation (Placement(transformation(extent={{-4,-30},{16,-10}})));

  Modelica.Blocks.Logical.And andHeaDemPri "Check if there even is demand"
    annotation (Placement(transformation(extent={{60,70},{80,90}})));
  Modelica.Blocks.Logical.Or orHeaDem
    "True if any of the two devices should turn on" annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-50,70})));

  Modelica.Blocks.Logical.Or orSecGen "Turn secondary device on if one is true"
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={40,-20})));
  Modelica.Blocks.Logical.Or orPriGen "Turn primary device on if one is true"
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={10,50})));
  Modelica.Blocks.Logical.And andHeaDemPri1 "Check if there even is demand"
    annotation (Placement(transformation(extent={{66,-10},{86,10}})));
equation
  connect(sub.u1, COP) annotation (Line(points={{-84,-14},{-94,-14},{-94,0},{
          -120,0}},
        color={0,0,127}));
  connect(sub.u2, COPMin) annotation (Line(points={{-84,-26},{-94,-26},{-94,-40},
          {-120,-40}},color={0,0,127}));
  connect(sub.y, lesCOPMin.u)
    annotation (Line(points={{-61,-20},{-42,-20}},
                                               color={0,0,127}));
  connect(lesCOPMin.y, not1.u)
    annotation (Line(points={{-19,-20},{-6,-20}},
                                             color={255,0,255}));
  connect(priGenOnSet, orHeaDem.u1) annotation (Line(points={{-120,80},{-70,80},
          {-70,70},{-62,70}}, color={255,0,255}));
  connect(orHeaDem.u2, secGenOnSet) annotation (Line(points={{-62,62},{-80,62},
          {-80,46},{-120,46}}, color={255,0,255}));
  connect(orPriGen.u1, priGenOnSet) annotation (Line(points={{-2,50},{-90,50},{
          -90,80},{-120,80}}, color={255,0,255}));
  connect(orSecGen.u2, not1.y) annotation (Line(points={{28,-28},{22,-28},{22,
          -20},{17,-20}}, color={255,0,255}));
  connect(orSecGen.u1, secGenOnSet) annotation (Line(points={{28,-20},{22,-20},
          {22,22},{-90,22},{-90,46},{-120,46}}, color={255,0,255}));
  connect(orSecGen.y, andHeaDemPri1.u2) annotation (Line(points={{51,-20},{60,
          -20},{60,-8},{64,-8}}, color={255,0,255}));
  connect(andHeaDemPri1.y, secGenOn)
    annotation (Line(points={{87,0},{110,0}}, color={255,0,255}));
  connect(andHeaDemPri.y, priGenOn)
    annotation (Line(points={{81,80},{110,80}}, color={255,0,255}));
  connect(andHeaDemPri1.u1, orHeaDem.y) annotation (Line(points={{64,0},{32,0},
          {32,70},{-39,70}}, color={255,0,255}));
  connect(orHeaDem.y, andHeaDemPri.u1) annotation (Line(points={{-39,70},{-30,
          70},{-30,80},{58,80}}, color={255,0,255}));
  connect(orPriGen.y, andHeaDemPri.u2) annotation (Line(points={{21,50},{50,50},
          {50,72},{58,72}}, color={255,0,255}));
  connect(orPriGen.u2, lesCOPMin.y) annotation (Line(points={{-2,42},{-8,42},{
          -8,-14},{-14,-14},{-14,-20},{-19,-20}}, color={255,0,255}));
  annotation (Diagram(coordinateSystem(extent={{-100,-100},{100,100}})), Icon(
        coordinateSystem(extent={{-100,-100},{80,100}})));
end COPBasedHybridControl;
