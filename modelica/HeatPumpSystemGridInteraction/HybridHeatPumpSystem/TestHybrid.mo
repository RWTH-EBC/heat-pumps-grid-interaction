within HeatPumpSystemGridInteraction.HybridHeatPumpSystem;
model TestHybrid
  extends HeatPumpSystemGridInteraction.HybridHeatPumpSystem.Hybrid(
    building(redeclare
        Buildings_Hybrid.EFH2010_standard.EFH2010_standard_DataBase.EFH2010_standard_SingleDwelling
        oneZoneParam),
    hydraulic(redeclare BESMod.Systems.Hydraulical.Transfer.UFHTransferSystem
        transfer(
        nHeaTra=1.3,
        redeclare
          BESMod.Systems.Hydraulical.Transfer.RecordsCollection.SteelRadiatorStandardPressureLossData
          parTra,
        redeclare
          BESMod.Systems.Hydraulical.Transfer.RecordsCollection.DefaultUFHData
          UFHParameters,
        redeclare BESMod.Systems.RecordsCollection.Movers.DefaultMover parPum)),
    parameterStudy(TBiv=271.739044189453),
    THyd_nominal=308.15,
    dTHyd_nominal=5,
    DHW(
      mDHW_flow_nominal=0.04,
      VDHWDay(displayUnit="l") = 0.1,
      tCrit=3600,
      QCrit=2.24,
      tableName="DHWCalc",
      fileName=Modelica.Utilities.Files.loadResource(
          "D:\01_Projekte\09_HybridWP\dhw_tappings\DHWCalc_78.txt")),
    systemParameters(filNamWea=Modelica.Utilities.Files.loadResource(
          "D://01_Projekte//09_HybridWP//01_Results//02_simulations//Hybrid_newbuildings//WeatherInputs//TRY2015_523845130645_Jahr.mos"),
        TOda_nominal=260.54999999999995),
    userProfiles(fileNameAbsGai=Modelica.Utilities.Files.loadResource(
          "D://01_Projekte//09_HybridWP//01_Results//02_simulations//Hybrid_newbuildings//custom_inputs//78.txt")));

end TestHybrid;
