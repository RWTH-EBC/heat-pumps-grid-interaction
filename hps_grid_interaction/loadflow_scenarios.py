import os


# Define scenario top/main folder:
SCENARIOS_MAIN_FOLDER = os.path.abspath(os.path.join(str(os.path.dirname(__file__)), "03_monte_carlo"))


# Iterate through all sub-folders and identify valid input Excel sheets for loadflow simulation:
LOADFLOW_SCENARIOS = list()
subfolders = [os.path.abspath(f.path) for f in os.scandir(SCENARIOS_MAIN_FOLDER) if f.is_dir()]
for directory in subfolders:
    for file in os.listdir(os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER, directory))):
        if ((not file.startswith("grid_plausibility")) and (not file.startswith("grid_statistics")) and
                (file.endswith(".xlsx"))):
            LOADFLOW_SCENARIOS.append(os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER, directory, file)))
