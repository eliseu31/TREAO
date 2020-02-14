from simulation.graph_manager import GraphManager
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if __name__ == '__main__':
    # array of files
    graphs_data = [('P0(G0)', 'graph.json', 'machines_specs.json',
                    ["M1", "M1", "M1", "M1", "M1", "M2", "M2", "M1"]),
                   ('P1(G1)', 'graph_3.json', 'machines_specs.json',
                    ["M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M1", "M2", "M2"]),
                   ('P2(G2)', 'graph_2.json', 'machines_specs_1.json',
                    ["M1", "M1", "M1", "M1", "M1", "M2", "M5", "M2", "M3", "M2", "M4", "M2", "M2"]),
                   ('P3(G2)', 'graph_2.json', 'machines_specs_1.json',
                    ["M2", "M2", "M2", "M1", "M1", "M1", "M1", "M1", "M4", "M5", "M1", "M3", "M2"])]
    table = [[], [], ["2/2", "2/2", "5/5", "6/5"]]
    rows = ['vertices', 'network/edges', 'macs/clusters']
    cost_data = dict()
    # configurations to validate
    root_path = os.path.dirname(sys.path[0])
    functions_path = os.path.join(root_path, 'resources', 'function_req.json')

    # for each file
    for name, file_name, machine_specs, placement in graphs_data:
        # graph path
        graph_path = os.path.join(root_path, 'resources', file_name)
        machines_path = os.path.join(root_path, 'resources', machine_specs)
        # creates the simulation graph
        simulation = GraphManager(graph_path, machines_path, functions_path)
        simulation.update_combination(placement)
        simulation.simulate_graph()
        # save the utility values
        cost_data[name] = simulation.read_utility_values()
        # save the number of nodes
        table[0].append(str(len(simulation.graph.nodes)))
        # save the number of edges
        large, short = simulation.update_edges()
        table[1].append('{0}/{1}'.format(len(large), len(large) + len(short)))

    fig = plt.figure(0, figsize=(4, 5))

    # plots the bar plots
    x = np.arange(len(cost_data)) + 0.6
    print(x)
    ram = [costs['ram'] for name, costs in cost_data.items()]
    plt.bar(x - 0.2, ram, width=0.1, color='b', align='center', label='ram')
    cpu = [costs['cpu'] for name, costs in cost_data.items()]
    plt.bar(x - 0.1, cpu, width=0.1, color='r', align='center', label='cpu')
    path = [costs['l_path'] for name, costs in cost_data.items()]
    plt.bar(x, ram, width=0.1, color='g', align='center', label='path')
    traffic = [costs['n_traffic'] for name, costs in cost_data.items()]
    plt.bar(x + 0.1, traffic, width=0.1, color='yellow', align='center', label='traffic')
    diff = [costs['s_mac'] for name, costs in cost_data.items()]
    plt.bar(x + 0.2, diff, width=0.1, color='brown', align='center', label='diff')
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
