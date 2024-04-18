import os

import numpy as np

# Define scenario top/main folder:
SCENARIOS_MAIN_FOLDER = os.path.abspath(os.path.join(str(os.path.dirname(__file__)), "03_monte_carlo"))


# Iterate through all sub-folders and identify valid input Excel sheets for loadflow simulation:
LOADFLOW_SCENARIOS = np.array([], dtype=str)
subfolders = [os.path.abspath(f.path) for f in os.scandir(SCENARIOS_MAIN_FOLDER) if f.is_dir()]
for subdirectory in subfolders:
    for file in os.listdir(os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER, subdirectory))):
        if ((not file.startswith("grid_plausibility")) and (not file.startswith("grid_statistics")) and
                (file.endswith(".xlsx"))):
            LOADFLOW_SCENARIOS = np.append(LOADFLOW_SCENARIOS, os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER,
                                                                                            subdirectory, file)))
    subsubfolders = [os.path.abspath(f.path) for f in
                     os.scandir(os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER, subdirectory))) if f.is_dir()]
    for subsubdirectory in subsubfolders:
        for file in os.listdir(os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER, subsubdirectory))):
            if ((not file.startswith("grid_plausibility")) and (not file.startswith("grid_statistics")) and
                    (file.endswith(".xlsx"))):
                LOADFLOW_SCENARIOS = np.append(LOADFLOW_SCENARIOS, os.path.abspath(os.path.join(SCENARIOS_MAIN_FOLDER,
                                                                                                subsubdirectory, file)))
