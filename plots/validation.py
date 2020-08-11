from simulation.graph_manager import GraphManager
from simulation.machine_data import MachinesManager
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if __name__ == '__main__':
    # array of files
    graphs_data = [('P0(G0)', 'graph_2.json', 'machines_specs.json',
                    ["M1", "M1", "M1", "M1", "M1", "M2", "M2", "M1"]),
                   ('P1(G1)', 'graph_3.json', 'machines_specs.json',
                    ["M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M2"]),
                   ('P2(G2)', 'graph.json', 'machines_specs_1.json',
                    ["M1", "M1", "M1", "M1", "M1", "M2", "M5", "M2", "M3", "M2", "M4", "M2", "M2"]),
                   ('P3(G2)', 'graph.json', 'machines_specs_1.json',
                    ["M2", "M2", "M2", "M1", "M1", "M1", "M1", "M1", "M4", "M5", "M1", "M3", "M2"])]
    table = [[], [], ["2/2", "2/2", "5/5", "6/5"]]
    rows = ['vertices', 'edges', 'clusters']
    cost_data = dict()
    err_data = dict()
    # configurations to validate
    root_path = os.path.dirname(sys.path[0])
    f_path = os.path.join(root_path, 'resources', 'profiling', 'task_req.json')
    fp_path = os.path.join(root_path, 'resources', 'profiling', 'task_req_prediction.json')
    # creates the machines manager
    md_ref = MachinesManager(f_path)
    md_pred = MachinesManager(fp_path)

    # for each file
    for name, file_name, machine_specs, placement in graphs_data:
        # graph path
        graph_path = os.path.join(root_path, 'resources', 'graphs', file_name)
        machines_path = os.path.join(root_path, 'resources', 'machine_specs', machine_specs)
        # creates the simulation graph
        simulation = GraphManager(graph_path, machines_path, md_ref)
        simulation.update_combination(placement)
        simulation.simulate_graph()
        # save the utility values
        cost_data[name] = simulation.read_utility_values()
        # save the number of nodes
        table[0].append(str(len(simulation.graph.nodes)))
        # save the number of edges
        large, short = simulation.update_edges()
        table[1].append('{0}/{1}'.format(len(large), len(large) + len(short)))

        # computes the hyper model predictions graph
        s_pred = GraphManager(graph_path, machines_path, md_pred)
        s_pred.update_combination(placement)
        s_pred.simulate_graph()
        # save the utility values
        uv_pred = s_pred.read_utility_values()
        err_data[name] = {key: abs(uv_pred[key]-cost_data[name][key])/2 for key in uv_pred.keys()}
        print(uv_pred)
        print(cost_data[name])

    fig = plt.figure(0, figsize=(4, 5))

    # plots the bar plots
    x = np.arange(len(cost_data)) + 0.6
    print(x)
    ram = [costs['ram'] for _, costs in cost_data.items()]
    err_ram = [err['ram'] for _, err in err_data.items()]
    plt.bar(x - 0.25, ram, width=0.1, color='b', align='center', label='ram', yerr=err_ram)
    cpu = [costs['cpu'] for _, costs in cost_data.items()]
    err_cpu = [err['cpu'] for _, err in err_data.items()]
    plt.bar(x - 0.15, cpu, width=0.1, color='r', align='center', label='cpu', yerr=err_cpu)
    path = [costs['l_path'] for _, costs in cost_data.items()]
    err_path = [err['l_path'] for _, err in err_data.items()]
    plt.bar(x - 0.05, ram, width=0.1, color='g', align='center', label='path', yerr=err_path)
    traffic = [costs['n_traffic'] for _, costs in cost_data.items()]
    err_traffic = [err['n_traffic'] for _, err in err_data.items()]
    plt.bar(x + 0.05, traffic, width=0.1, color='yellow', align='center', label='traffic', yerr=err_traffic)
    diff = [costs['s_mac'] for _, costs in cost_data.items()]
    err_diff = [err['s_mac'] for _, err in err_data.items()]
    plt.bar(x + 0.15, diff, width=0.1, color='brown', align='center', label='diff', yerr=err_diff)
    clusters = [costs['clusters'] for _, costs in cost_data.items()]
    err_clusters = [err['clusters'] for _, err in err_data.items()]
    plt.bar(x + 0.25, clusters, width=0.1, color='lightblue', align='center', label='cluster', yerr=err_clusters)

    plt.ylim([0, 0.4])
    plt.ylabel("Normalized Costs")

    # plots the table
    plt.table(cellText=table, rowLabels=rows, colLabels=list(cost_data.keys()), loc='bottom', cellLoc='center')
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.xticks([])

    plt.legend()
    plt.title("Metrics Comparison")
    plt.tight_layout()
    plt.show()
