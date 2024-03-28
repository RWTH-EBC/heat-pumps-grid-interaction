import pathlib

PROJECT_FOLDER = pathlib.Path(r"D:\fwu\01_Projekte\01_HybridWP")
RESULTS_BES_FOLDER = PROJECT_FOLDER.joinpath("01_results", "02_simulations")
RESULTS_MONTE_CARLO_FOLDER = PROJECT_FOLDER.joinpath("01_results", "03_monte_carlo")
RESULTS_GRID_FOLDER = PROJECT_FOLDER.joinpath("01_results", "04_lastfluss")
BESMOD_PATH = pathlib.Path(r"D:\fwu\04_git\BESMod\startup.mos")
REPO_ROOT = pathlib.Path(__file__).absolute().parents[1]
KERBER_NETZ_XLSX = REPO_ROOT.joinpath("Kerber_Vorstadtnetz.xlsx")
DATA_PATH = REPO_ROOT.joinpath("data")
PLOTS_PATH = REPO_ROOT.joinpath("plots")
MODELICA_PATH = REPO_ROOT.joinpath("modelica", "HeatPumpSystemGridInteraction", "package.mo")
E_MOBILITY_DATA = PROJECT_FOLDER.joinpath("time_series_data", "e_mobility")
HOUSEHOLD_DATA = PROJECT_FOLDER.joinpath("time_series_data", "household")
