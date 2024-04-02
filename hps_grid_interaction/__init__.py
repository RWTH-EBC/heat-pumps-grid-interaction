import json
import os.path
import pathlib


REPO_ROOT = pathlib.Path(__file__).absolute().parents[1]
PC_SPECIFIC_SETTING_PATH = REPO_ROOT.joinpath("pc_specific_settings.json")
if os.path.exists(PC_SPECIFIC_SETTING_PATH):
    with open(PC_SPECIFIC_SETTING_PATH, "r") as file:
        PC_SPECIFIC_SETTINGS = json.load(file)
    PROJECT_FOLDER = pathlib.Path(PC_SPECIFIC_SETTINGS["PROJECT_FOLDER"])
    BESMOD_PATH = pathlib.Path((PC_SPECIFIC_SETTINGS["BESMOD_PATH"]))
else:
    PROJECT_FOLDER = pathlib.Path(r"D:\01_Projekte\09_HybridWP")
    BESMOD_PATH = pathlib.Path(r"D:\04_git\BESMod\startup.mos")

RESULTS_BES_FOLDER = PROJECT_FOLDER.joinpath("01_results", "02_simulations")
RESULTS_MONTE_CARLO_FOLDER = PROJECT_FOLDER.joinpath("01_results", "03_monte_carlo")
RESULTS_GRID_FOLDER = PROJECT_FOLDER.joinpath("01_results", "04_lastfluss")
KERBER_NETZ_XLSX = REPO_ROOT.joinpath("Kerber_Vorstadtnetz.xlsx")
DATA_PATH = REPO_ROOT.joinpath("data")
PLOTS_PATH = REPO_ROOT.joinpath("plots")
MODELICA_PATH = REPO_ROOT.joinpath("modelica", "HeatPumpSystemGridInteraction", "package.mo")
E_MOBILITY_DATA = PROJECT_FOLDER.joinpath("time_series_data", "e_mobility")
HOUSEHOLD_DATA = PROJECT_FOLDER.joinpath("time_series_data", "household")
