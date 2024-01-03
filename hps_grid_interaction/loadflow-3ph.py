import os
import math
import pathlib
import multiprocessing as mp
import openpyxl
import numpy as np
import pandas as pd

# We are using pandapower 2.13.1
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
import pandapower.plotting as plotting
from pandapower.timeseries.data_sources.frame_data import DFData


def run_lastfluss_simulation(
        trafo_kva: list,
        input_path: pathlib.Path,
        cos_phi: float = 0.95,
        save_path: pathlib.Path = None,
        n_cpu: int = 1,
        sheet_name: str = "lastfluss"
):
    if save_path is None:
        save_path = input_path
    files = [input_path.joinpath(file) for file in os.listdir(input_path) if file.endswith(".xlsx")]
    simulations_to_run = []
    for kva in trafo_kva:
        for file in files:
            case = file.stem
            if cos_phi != 0.95:
                save_path_case = save_path.joinpath(f"3-ph-{(int(cos_phi*100))}", f"{case}_{int(kva)}")
            else:
                save_path_case = save_path.joinpath("3-ph", f"{case}_{int(kva)}")
            if save_path_case.exists():
                print(f"Won't run sheet {save_path_case.name}, folder already exists.")
                continue
            if save_path_case.name.startswith("Mon_"):  # TODO: Revert
                print("Skipping Simulation of monovalent cases for GEG-update:", save_path_case)
                continue
            workbook = openpyxl.load_workbook(file)
            ws = workbook[sheet_name]
            simulations_to_run.append(dict(
                worksheet=ws, kva=kva, case=case, cos_phi=cos_phi,
                save_path=save_path_case,
                with_plot=False
            ))
    if n_cpu == 1:
        for simulation_to_run in simulations_to_run:
            run_single_worksheet(simulation_to_run)
    else:
        pool = mp.Pool(processes=n_cpu)
        results = []
        for result in pool.imap(run_single_worksheet, simulations_to_run):
            results.append(result)


def run_single_worksheet(
    simulation_to_run: dict
):
    worksheet = simulation_to_run["worksheet"]
    kva = simulation_to_run["kva"]
    case = simulation_to_run["case"]
    cos_phi = simulation_to_run["cos_phi"]
    save_path = simulation_to_run["save_path"]
    with_plot = simulation_to_run["with_plot"]
    print("Conducting the load flow simulation for scenario " + str(case) + " with trafo size " +
          str(int(kva)) + "kVA.")

    connection_points = np.empty(len(worksheet["A"]) - 1, dtype=object)
    elec_demand_files = np.empty(len(worksheet["A"]) - 1, dtype=object)
    hp_demand_files = np.empty(len(worksheet["A"]) - 1, dtype=object)
    ev_demand_files = np.empty(len(worksheet["A"]) - 1, dtype=object)

    for i in range(len(worksheet["A"]) - 1):
        connection_points[i] = "loadbus_" + str(worksheet["D"][i + 1].value).replace("-", "_") + ""
        elec_demand_files[i] = str(worksheet["H"][i + 1].value)
        hp_demand_files[i] = str(worksheet["I"][i + 1].value)
        ev_demand_files[i] = str(worksheet["J"][i + 1].value)

    elec_p_demand_timeseries = np.empty_like(elec_demand_files)
    hp_p_demand_timeseries = np.empty_like(hp_demand_files)
    ev_p_demand_timeseries = np.empty_like(ev_demand_files)

    elec_q_demand_timeseries = np.empty_like(elec_demand_files)
    hp_q_demand_timeseries = np.empty_like(hp_demand_files)
    ev_q_demand_timeseries = np.empty_like(ev_demand_files)

    # load time series CSV files with loads (this will take a while!)
    for i in range(len(worksheet["A"]) - 1):
        print("Preparing input " + str(i + 1) + "/" + str(len(worksheet["A"])) + ".")

        elec_p_demand_timeseries[i] = pd.read_csv(os.path.abspath(elec_demand_files[i]), usecols=[1],
                                                  names=["power"], header=None)
        hp_p_demand_timeseries[i] = pd.read_csv(os.path.abspath(hp_demand_files[i]), usecols=[1],
                                                names=["power"], header=None)
        ev_p_demand_timeseries[i] = pd.read_csv(os.path.abspath(ev_demand_files[i]), usecols=[1],
                                                names=["power"], header=None)

        elec_p_demand_timeseries[i].drop(elec_p_demand_timeseries[i].head(1).index, inplace=True)
        hp_p_demand_timeseries[i].drop(hp_p_demand_timeseries[i].head(1).index, inplace=True)
        hp_p_demand_timeseries[i].drop(hp_p_demand_timeseries[i].tail(1).index, inplace=True)
        ev_p_demand_timeseries[i].drop(ev_p_demand_timeseries[i].head(1).index, inplace=True)

        elec_p_demand_timeseries[i].power = elec_p_demand_timeseries[i].power.astype(float)
        hp_p_demand_timeseries[i].power = hp_p_demand_timeseries[i].power.astype(float)
        ev_p_demand_timeseries[i].power = ev_p_demand_timeseries[i].power.astype(float)

        elec_q_demand_timeseries[i] = elec_p_demand_timeseries[i].copy()
        hp_q_demand_timeseries[i] = hp_p_demand_timeseries[i].copy()
        ev_q_demand_timeseries[i] = ev_p_demand_timeseries[i].copy()

        elec_q_demand_timeseries[i]["power"] = \
            np.sqrt((elec_q_demand_timeseries[i]["power"] / cos_phi) ** 2 -
                    elec_q_demand_timeseries[i]["power"] ** 2)

        hp_q_demand_timeseries[i]["power"] = \
            np.sqrt((hp_q_demand_timeseries[i]["power"] / cos_phi) ** 2 -
                    hp_q_demand_timeseries[i]["power"] ** 2)

        ev_q_demand_timeseries[i]["power"] = \
            np.sqrt((ev_q_demand_timeseries[i]["power"] / cos_phi) ** 2 -
                    ev_q_demand_timeseries[i]["power"] ** 2)

    # load a pandapower network
    net = nw.create_kerber_vorstadtnetz_kabel_1()
    net.trafo.sn_mva = kva / 1000.0 / 3.0
    net.trafo.tap_min = 0
    net.trafo.tap_max = 0

    # assemble the loads from the different CSV files and make it fit to the pandapower's Kerber grid model
    busdict = net.bus.to_dict()
    loadbusdict = dict()
    for key, value in busdict["name"].items():
        if str(value).startswith("loadbus"):
            loadbusdict[str(value)] = int(net.load.bus[net.load.bus == key].index[0])

    df_active = pd.DataFrame()
    df_reactive = pd.DataFrame()
    for i in range(len(worksheet["A"]) - 1):
        load_id = loadbusdict[connection_points[i]]
        df_active[load_id] = elec_p_demand_timeseries[load_id]["power"] + \
                             hp_p_demand_timeseries[load_id]["power"] + \
                             ev_p_demand_timeseries[load_id]["power"]
        df_active[load_id] = df_active[load_id].multiply(1.0 / 3.0)
        df_active[load_id] = df_active[load_id].multiply(1.0 / 1000.0)
        df_reactive[load_id] = elec_q_demand_timeseries[load_id]["power"] + \
                               hp_q_demand_timeseries[load_id]["power"] + \
                               ev_q_demand_timeseries[load_id]["power"]
        df_reactive[load_id] = df_reactive[load_id].multiply(1.0 / 3.0)
        df_reactive[load_id] = df_reactive[load_id].multiply(1.0 / 1000.0)
    df_active = df_active.reindex(sorted(df_active.columns), axis=1)
    df_active.index = range(0, len(df_active))
    df_reactive = df_reactive.reindex(sorted(df_reactive.columns), axis=1)
    df_reactive.index = range(0, len(df_reactive))

    # update the loads of the pandapower Kerber grid
    ds_active = DFData(df_active)
    active_load = control.ConstControl(net, element='load', element_index=net.load.index,
                                       variable='p_mw', data_source=ds_active, profile_name=net.load.index)

    ds_reactive = DFData(df_reactive)
    reactive_load = control.ConstControl(net, element='load', element_index=net.load.index,
                                         variable='q_mvar', data_source=ds_reactive, profile_name=net.load.index)

    # initialising the outputwriter to save data to excel files in the current folder
    # You can change this to .json, .csv, or .pickle as well
    ow = timeseries.OutputWriter(net,
                                 output_path=str(save_path),
                                 output_file_type=".xlsx")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')

    # starting the timeseries simulation for one year in 15 min values.
    print("Performing the load flow simulation.")
    timeseries.run_timeseries(net)

    print("Load flow simulation successful.")
    if with_plot:
        # Finally and if desired, create an HTML plot
        plotting.to_html(net, "kerber.html")


if __name__ == '__main__':
    from hps_grid_interaction import RESULTS_GRID_FOLDER
    run_lastfluss_simulation(
        trafo_kva=[400, 630.0, 1000, 2000],
        input_file=RESULTS_GRID_FOLDER.joinpath("LastflussSimulationenGEGBiv.xlsx"),
        n_cpu=2,
        cos_phi=0.95
    )
