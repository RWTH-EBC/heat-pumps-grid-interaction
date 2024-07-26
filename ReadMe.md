# What is this?

Repository with method and scripts for the publication titled
"Impact of hybrid heat pump shares and building retrofit rates on load penetration in German low-voltage grids", submitted to Applied Energy.
TODO: Preprint.

# How to use?

As requirements, you have to install the requirements with specified version in `requirements.txt` using
```
pip install -r requirements.txt
```
To simulate the building energy system models, you will need BESMod 
and Dymola. We used Dymola2023x for the simulations.
To install BESMod, follow the ReadMe here: 
We used the commit `9e6f50e8f0e0436b8c1ed5d28d585ddd39ef8643` to create the simulation results.
However, all relevant simulation results are exported as .csv and uploaded to the 
research data repository # TODO

1. For BES-simulations, run `run_studies.py`
2. To perform monte-carlo simulations, run `hps_grid_interaction/monte_carlo/monte_carlo.py`
3. To perform grid simulations, run the files `hps_grid_interaction/loadflow_3ph_mpi.py`
4. To generate plots as in the publication, run `hps_grid_interaction/plotting/plot_loadflow.py`

Please note that all function were only tested on a Windows machine with python 3.10.
If you have issues reproducing results or calling functions, please raise an issue or contact the authors.
