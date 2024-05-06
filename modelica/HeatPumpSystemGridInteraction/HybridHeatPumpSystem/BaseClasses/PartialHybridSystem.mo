within HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses;
partial model PartialHybridSystem "Partial bivalent heat pump system"
  extends HeatPumpSystemGridInteraction.BaseClasses.PartialSystem2D(
    final scalingFactor=hydraulic.generation.parHeaPum.scalingFactor,
    redeclare BESMod.Systems.Electrical.ElectricalSystem electrical(
      redeclare
        HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.PVSystemWithElectricCar
        generation(
        final f_design=fill(1, electrical.generation.numGenUnits),
        useTwoRoo=false,
        tilAllMod=0.5235987755983,
        redeclare model CellTemperature =
            AixLib.Electrical.PVSystem.BaseClasses.CellTemperatureMountingContactToGround,
        redeclare AixLib.DataBase.SolarElectric.QPlusBFRG41285 pVParameters,
        lat=weaDat.lat,
        lon=weaDat.lon,
        alt=weaDat.alt,
        timZon=weaDat.timZon,
        ARoo=building.ARoo/2,
        use_eMob=use_eMob,
        fileNameEMob=fileNameEMob),
      redeclare BESMod.Systems.Electrical.Distribution.BatterySystemSimple
        distribution(nBat=2, redeclare
          BuildingSystems.Technologies.ElectricalStorages.Data.LithiumIon.LithiumIonViessmann
          batteryParameters),
      redeclare BESMod.Systems.Electrical.Transfer.NoElectricalTransfer transfer,
      redeclare BESMod.Systems.Electrical.Control.NoControl control),
    redeclare HybridHeatPumpSystem.BaseClasses.CustomTEASERThermalZone building(
        thermalZone(internalGainsMode=2)),
    redeclare BESMod.Systems.Control.NoControl control,
    redeclare BESMod.Systems.Hydraulical.HydraulicSystem hydraulic(redeclare
        HybridHeatPumpSystem.BaseClasses.HeatPumpAndElectricHeater generation(
        redeclare
          BESMod.Systems.Hydraulical.Generation.RecordsCollection.DefaultHP
          parHeaPum(
          genDesTyp=genDesTyp,
          TBiv=parameterStudy.TBiv,
          QPriAtTOdaNom_flow_nominal=QPriAtTOdaNom_flow_nominal,
          scalingFactor=hydraulic.generation.parHeaPum.QPri_flow_nominal/
              QHeaPumBiv_flow,
          mEva_flow_nominal=hydraulic.generation.m_flow_nominal[1]*4,
          dpCon_nominal=0,
          dpEva_nominal=0,
          use_refIne=false,
          refIneFre_constant=0),
        redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum,
        redeclare
          BESMod.Systems.RecordsCollection.TemperatureSensors.DefaultSensor
          parTemSen(transferHeat=true),
        use_eleHea=use_eleHea,
        redeclare
          BESMod.Systems.Hydraulical.Generation.RecordsCollection.DefaultElectricHeater
          parEleHea,
        redeclare HeatPumpSystemGridInteraction.RecordsCollection.VitoCal250
          dataTable)),
    redeclare BESMod.Systems.Demand.DHW.DHWCalc DHW(redeclare
        BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum, redeclare
        BESMod.Systems.Demand.DHW.TappingProfiles.calcmFlowEquDynamic calcmFlow),
    redeclare
      HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.CustomInputs
      userProfiles,
    redeclare
      HeatPumpSystemGridInteraction.HybridHeatPumpSystem.BaseClasses.ParameterStudy
      parameterStudy);

  parameter Modelica.Units.SI.TemperatureDifference dTAddHeaCur=max(THeaCur_nominal - THyd_nominal, 0)
    "Constant offset of ideal heating curve";
  parameter Modelica.Units.SI.Temperature THeaCur_nominal=THyd_nominal
    "Nominal set temperature of heating curve";
  parameter Boolean use_eleHea=true "=false to disable the electric heater";
  parameter BESMod.Systems.Hydraulical.Generation.Types.GenerationDesign
    genDesTyp=BESMod.Systems.Hydraulical.Generation.Types.GenerationDesign.BivalentPartParallel
    "Type of generation system design";
  parameter Boolean use_eMob=false "= true to activate e mobility";
  parameter String fileNameEMob=Modelica.Utilities.Files.loadResource(
      "modelica://HeatPumpSystemGridInteraction/HybridHeatPumpSystem/CustomInputs.txt")
    "File where data for e mobility is stored";
  annotation (experiment(
      StopTime=31536000,
      Interval=600,
      __Dymola_Algorithm="Dassl"));
end PartialHybridSystem;
