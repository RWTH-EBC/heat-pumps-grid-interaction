within HeatPumpSystemGridInteraction.HybridHeatPumpSystem;
model TestMonovalent
  extends Monovalent(building(redeclare Buildings_Monovalent.MFH1980_standard_10.MFH1980_standard_10_DataBase.MFH1980_standard_10_SingleDwelling oneZoneParam),
  parameterStudy(
      TBiv=systemParameters.TOda_nominal,
      THyd_nominal=343.15,
    dTHyd_nominal=15),
  DHW(redeclare BESMod.Systems.Demand.DHW.RecordsCollection.NoDHW DHWProfile),
  systemParameters(
    filNamWea=Modelica.Utilities.Files.loadResource("D://01_Projekte//09_HybridWP//01_Results//02_simulations//Monovalent//WeatherInputs//TRY2015_523845130645_Jahr.mos"),
    TOda_nominal=260.54999999999995),
  userProfiles(fileNameAbsGai=Modelica.Utilities.Files.loadResource("D://01_Projekte//09_HybridWP//01_Results//02_simulations//custom_inputs//MFH1980_standard_10.txt")));
end TestMonovalent;
