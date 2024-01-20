import json
import os
import uuid
import shutil
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
        loadflow_scenarios: list,
        cos_phi: float = 0.95,
        n_cpu: int = 1,
        sheet_name: str = "lastfluss"
):
    count = 0
    simulations_to_run = []
    for kva in trafo_kva:
        for file in loadflow_scenarios:
            count += 1
            path = os.path.dirname(file)
            filename = os.path.basename(file)
            case_name = os.path.abspath(
                os.path.join(path,
                             "results_" + str(str(os.path.splitext(filename)[0]) + "_3-ph_" + str(int(kva)) + ".json"))
            )
            if os.path.exists(case_name):
                print("Won't run simulation " + str(filename) + ", solution JSON file already exists.")
                continue
            workbook = openpyxl.load_workbook(file)
            ws = workbook[sheet_name]
            simulations_to_run.append(dict(
                worksheet=ws, kva=kva, case=str(filename), cos_phi=cos_phi,
                save_path=case_name, with_plot=False, case_no=count)
            )
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

    random_str = str(uuid.uuid4())

    connection_points = np.empty(len(worksheet["A"]) - 1, dtype=object)
    elec_demand_files = np.empty(len(worksheet["A"]) - 1, dtype=object)

    for i in range(len(worksheet["A"]) - 1):
        connection_points[i] = "loadbus_" + str(worksheet["D"][i + 1].value).replace("-", "_") + ""
        elec_demand_files[i] = str(worksheet["I"][i + 1].value)

    elec_p_demand_timeseries = np.empty_like(elec_demand_files)
    elec_q_demand_timeseries = np.empty_like(elec_demand_files)

    # load time series CSV files with loads (this will take a while!)
    for i in range(len(worksheet["A"]) - 1):
        print("Preparing input " + str(i + 1) + "/" + str(len(worksheet["A"])-1) + ".")

        elec_p_demand_timeseries[i] = pd.read_csv(
            os.path.abspath(os.path.join(str(os.path.dirname(save_path)), str(elec_demand_files[i])).
                            replace("\\","/")), usecols=[1], names=["power"], header=None
        )

        elec_p_demand_timeseries[i].drop(elec_p_demand_timeseries[i].head(1).index, inplace=True)
        elec_p_demand_timeseries[i].power = elec_p_demand_timeseries[i].power.astype(float)

        elec_q_demand_timeseries[i] = elec_p_demand_timeseries[i].copy()
        elec_q_demand_timeseries[i]["power"] = \
            np.sqrt((elec_q_demand_timeseries[i]["power"] / cos_phi) ** 2 -
                    elec_q_demand_timeseries[i]["power"] ** 2)

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
        df_active[load_id] = elec_p_demand_timeseries[load_id]["power"]
        df_active[load_id] = df_active[load_id].multiply(1.0 / 3.0)
        df_active[load_id] = df_active[load_id].multiply(1.0 / 1000.0)
        df_reactive[load_id] = elec_q_demand_timeseries[load_id]["power"]
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

    # initialising the outputwriter to save data to JSON files in the specified folder(s)
    # You can change this to .json, .csv, or .pickle as well
    ow = timeseries.OutputWriter(net,
                                 output_path=os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str)),
                                 output_file_type=".json")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')

    # starting the timeseries simulation for one year in 15 min values.
    print("Performing the load flow simulation.")
    timeseries.run_timeseries(net, numba=False)

    print("Load flow simulation successful.")
    if with_plot:
        # Finally and if desired, create an HTML plot
        plotting.to_html(net, "kerber.html")

    print("Writing results.")
    p_trafo = np.array([0.0 for i in range(len(df_active))])
    q_trafo = np.array([0.0 for i in range(len(df_active))])
    vm_pu_min = np.array([np.infty for i in range(len(df_active))])
    max_line_loading = np.array([0.0 for i in range(len(df_active))])

    with open(os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str, "res_bus",
                                           "p_mw.json")), 'r') as j:
        p_mw = json.loads(j.read())
    count = 0
    for key, value in p_mw["0"].items():
        p_trafo[count] = (-1.0) * value * 1000.0
        count += 1

    with open(os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str, "res_bus",
                                           "q_mvar.json")), 'r') as j:
        q_mvar = json.loads(j.read())
    count = 0
    for key, value in q_mvar["0"].items():
        q_trafo[count] = (-1.0) * value * 1000.0
        count += 1

    s_trafo = np.array([np.sqrt(p_trafo[i]**2 + q_trafo[i]**2) for i in range(len(df_active))])

    with open(os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str, "res_bus",
                                           "vm_pu.json")), 'r') as j:
        vm_pu = json.loads(j.read())
    for key_outer, value_outer in vm_pu.items():
        count = 0
        for key_inner, value_inner in vm_pu[key_outer].items():
            vm_pu_min[count] = min(vm_pu_min[count], value_inner)
            count += 1

    with open(os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str, "res_line",
                                           "loading_percent.json")), 'r') as j:
        loading_percent = json.loads(j.read())
    for key_outer, value_outer in loading_percent.items():
        count = 0
        for key_inner, value_inner in loading_percent[key_outer].items():
            max_line_loading[count] = max(max_line_loading[count], value_inner)
            count += 1

    results_dict = dict(p_trafo=list(p_trafo), q_trafo=list(q_trafo), s_trafo=list(s_trafo),
                        vm_pu_min=list(vm_pu_min), max_line_loading=list(max_line_loading))
    with open(os.path.abspath(
            os.path.join(str(os.path.dirname(save_path)),
                         "results_" + str(case).split(".")[0] + "_3-ph_" + str(int(kva)) + ".json")), 'w') as k:
        json.dump(results_dict, k)

    # Finally remove all temporary files:
    shutil.rmtree(os.path.abspath(os.path.join(str(os.path.dirname(save_path)), random_str)))
    print("Success.")


if __name__ == '__main__':
    from loadflow_scenarios import LOADFLOW_SCENARIOS

    run_lastfluss_simulation(
        trafo_kva=[1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        loadflow_scenarios=LOADFLOW_SCENARIOS,
        n_cpu=1,
        cos_phi=0.95
    )