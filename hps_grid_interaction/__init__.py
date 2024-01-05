import pathlib

PROJECT_FOLDER = pathlib.Path(r"D:\01_Projekte\09_HybridWP")
RESULTS_BES_FOLDER = PROJECT_FOLDER.joinpath("01_results", "02_simulations")
RESULTS_GRID_FOLDER = PROJECT_FOLDER.joinpath("01_results", "03_lastfluss")
BESMOD_PATH = pathlib.Path(r"D:\04_git\BESMod\startup.mos")
KERBER_NETZ_XLSX = pathlib.Path(__file__).absolute().parents[1].joinpath("Kerber_Vorstadtnetz.xlsx")

E_MOBILITY_DATA = PROJECT_FOLDER.joinpath("time_series_data", "e_mobility")
HOUSEHOLD_DATA = PROJECT_FOLDER.joinpath("time_series_data", "household")
