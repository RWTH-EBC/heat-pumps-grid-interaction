within HeatPumpSystemGridInteraction.HybridHeatPumpSystem;
model Hybrid
  "Bivalent Heat Pump System with boiler integration after buffer tank without DHW support"
  extends HybridHeatPumpSystem.BaseClasses.PartialHybridSystem(
    genDesTyp=BESMod.Systems.Hydraulical.Generation.Types.GenerationDesign.BivalentAlternativ,
    use_eleHea=false,
    redeclare BESMod.Systems.Hydraulical.HydraulicSystem hydraulic(redeclare
        HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.HybridHeatPumpSystemCOPBased
        control(
        redeclare
          BESMod.Systems.Hydraulical.Control.Components.ThermostaticValveController.ThermostaticValvePIControlled
          valCtrl,
        dTHysBui=10,
        dTHysDHW=10,
        meaValPriGen=BESMod.Systems.Hydraulical.Control.Components.BaseClasses.MeasuredValue.GenerationSupplyTemperature,

        redeclare model DHWHysteresis =
            BESMod.Systems.Hydraulical.Control.Components.BivalentOnOffControllers.AlternativeBivalent
            (T_biv=parameterStudy.TBiv),
        redeclare model BuildingHysteresis =
            BESMod.Systems.Hydraulical.Control.Components.BivalentOnOffControllers.AlternativeBivalent
            (T_biv=parameterStudy.TBiv),
        redeclare model DHWSetTemperature =
            BESMod.Systems.Hydraulical.Control.Components.DHWSetControl.ConstTSet_DHW,

        redeclare model SummerMode =
            BESMod.Systems.Hydraulical.Control.Components.SummerMode.No,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.BasicHeatPumpPI
          parPIDHeaPum,
        TBiv=parameterStudy.TBiv,
        boiInGeneration=false,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.DefaultSafetyControl
          safetyControl,
        TCutOff=parameterStudy.TCutOff,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.BasicBoilerPI
          parPIDBoi,
        redeclare
          HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.NoCOPBasedHybridControl
          boiInHybSys), redeclare
        BESMod.Systems.Hydraulical.Distribution.TwoStoragesBoilerWithDHW
        distribution(
        redeclare
          BESMod.Systems.RecordsCollection.TemperatureSensors.DefaultSensor
          parTemSen,
        redeclare BESMod.Systems.RecordsCollection.Valves.DefaultThreeWayValve
          parThrWayVal,
        redeclare
          BESMod.Systems.Hydraulical.Distribution.RecordsCollection.BufferStorage.DefaultDetailedStorage
          parStoBuf(VPerQ_flow=parameterStudy.VPerQFlow),
        redeclare
          BESMod.Systems.Hydraulical.Distribution.RecordsCollection.BufferStorage.DefaultDetailedStorage
          parStoDHW(dTLoadingHC1=10),
        dTBoiDHWLoa=10,
        redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum,
        redeclare BESMod.Systems.RecordsCollection.Valves.DefaultThreeWayValve
          parThrWayValBoi,
        redeclare
          BESMod.Systems.Hydraulical.Distribution.RecordsCollection.BufferStorage.DefaultDetailedStorage
          parHydSep)));

  extends Modelica.Icons.Example;

  annotation (experiment(
      StopTime=31536000,
      Interval=600,
      __Dymola_Algorithm="Dassl"));
end Hybrid;
