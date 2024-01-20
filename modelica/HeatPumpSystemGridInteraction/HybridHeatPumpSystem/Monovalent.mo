within HeatPumpSystemGridInteraction.HybridHeatPumpSystem;
model Monovalent "Monovalent heat pump system"
  extends
    HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.PartialHybridSystem(
    genDesTyp=BESMod.Systems.Hydraulical.Generation.Types.GenerationDesign.BivalentParallel,
    use_eleHea=true,
    redeclare BESMod.Systems.Hydraulical.HydraulicSystem hydraulic(redeclare
        BESMod.Systems.Hydraulical.Control.MonoenergeticHeatPumpSystem control(
        redeclare
          BESMod.Systems.Hydraulical.Control.Components.ThermostaticValveController.ThermostaticValvePIControlled
          valCtrl,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.ThermostaticValveDataDefinition
          parTheVal,
        dTHysBui=10,
        dTHysDHW=10,
        redeclare model DHWHysteresis =
            BESMod.Systems.Hydraulical.Control.Components.BivalentOnOffControllers.ConstantHysteresisTimeBasedHeatingRod,
        redeclare model BuildingHysteresis =
            BESMod.Systems.Hydraulical.Control.Components.BivalentOnOffControllers.ConstantHysteresisTimeBasedHeatingRod,
        redeclare model SummerMode =
            BESMod.Systems.Hydraulical.Control.Components.SummerMode.No,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.BasicHeatPumpPI
          parPIDHeaPum,
        redeclare
          BESMod.Systems.Hydraulical.Control.RecordsCollection.DefaultSafetyControl
          safetyControl),
             redeclare
        BESMod.Systems.Hydraulical.Distribution.TwoStoDetailedDirectLoading
        distribution(
        redeclare
          BESMod.Systems.RecordsCollection.TemperatureSensors.DefaultSensor
          parTemSen,
        redeclare
          BESMod.Systems.Hydraulical.Distribution.RecordsCollection.BufferStorage.DefaultDetailedStorage
          parStoBuf(VPerQ_flow=parameterStudy.VPerQFlow),
        redeclare
          BESMod.Systems.Hydraulical.Distribution.RecordsCollection.BufferStorage.DefaultDetailedStorage
          parStoDHW(dTLoadingHC1=10),
        redeclare BESMod.Systems.RecordsCollection.Valves.DefaultThreeWayValve
          parThrWayVal)));

  extends Modelica.Icons.Example;

  annotation (experiment(
      StopTime=31536000,
      Interval=600,
      __Dymola_Algorithm="Dassl"));
end Monovalent;
