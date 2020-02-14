import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from simulation.graph_manager import GraphManager
from optimization.genetic_algorithm import GeneticAlgorithm


if __name__ == '__main__':
    # get the resources paths
    root_path = os.path.dirname(sys.path[0])
    # default relative paths
    graph_rpath = os.path.join('resources', 'graph_3.json')
    machines_rpath = os.path.join('resources', 'machines_specs.json')
    functions_rpath = os.path.join('resources', 'function_req.json')
    results_rpath = os.path.join('resources', 'simulation_results.json')
    graph_plt_rpath = os.path.join('resources', 'plot_results.png')
    # define default json
    ga_param = {"initial_population": 100, "selected_individuals": 20, "population": 50, "mutations": 20}
    policies = {"ram": 1, "cpu": 1, "l_path": 1, "n_traffic": 1, "s_mac": 1}

    parser = argparse.ArgumentParser()
    # parses the path arguments
    parser.add_argument('-g', metavar='graph', nargs=1, help="")
    parser.add_argument('-f', metavar='functions_req', nargs=1, help="")
    parser.add_argument('-m', metavar='machines_specs', nargs=1, help="")
    parser.add_argument('-p', metavar='profile_data', nargs=1, help="")
    parser.add_argument('-r', metavar='output_results', nargs=1, help="")
    parser.add_argument('-t', metavar='output_plot', nargs=1, help="")
    # options to execute
    parser.add_argument('-o', metavar='optimization_parameters', nargs=1, help="")
    parser.add_argument('-d', metavar='machine_distributions', nargs=1, help="")
    args = parser.parse_args()

    # checks the optional variables
    graph_rpath = args.g[0] if args.g else graph_rpath
    functions_rpath = args.f[0] if args.f else functions_rpath
    machines_rpath = args.m[0] if args.m else machines_rpath
    results_rpath = args.r[0] if args.r else results_rpath
    graph_plt_rpath = args.t[0] if args.t else graph_plt_rpath
    # configures the profiling path
    profiling_path = os.path.join(root_path, args.p[0]) if args.p else None

    # absolute path to each input variable
    graph_path = os.path.join(root_path, graph_rpath)
    machines_path = os.path.join(root_path, machines_rpath)
    functions_path = os.path.join(root_path, functions_rpath)
    results_path = os.path.join(root_path, results_rpath)
    graph_plt_path = os.path.join(root_path, graph_plt_rpath)

    # creates the simulation graph
    simulation = GraphManager(graph_path, machines_path, functions_path, profiling_path)

    # checks if visualizes the profiled data
    if args.d:
        # plots the machine distribution
        simulation.mm.machines_data[args.d[0]].plot_histogram()
        # exits the program
        sys.exit()

    # starts the simulation
    simulation.init_simulation()

    # receives the optimization parameters
    if args.o:
        param_path = os.path.join(root_path, args.o[0])
        # opens the file
        with open(param_path, 'r') as f:
            param = json.loads(f.read())
            ga_param = param['ga']
            policies = param['policies']

        # starts the optimization
        ga = GeneticAlgorithm(simulation, policies)
        ga.create_population(ga_param['initial_population'])
        ga.evaluate()
        ga.rank_selection(ga_param['selected_individuals'])

        graph_alive = True
        while graph_alive:
            ga.create_new_population(ga_param['population'], n_mutations=ga_param['mutations'])
            ga.evaluate()
            ga.rank_selection(ga_param['selected_individuals'])
            # updates the GUI
            stronger_individual, _ = ga.select_stronger()
            simulation.update_combination(stronger_individual)
            simulation.simulate_graph()
            graph_alive = simulation.flush_plot(float(ga_param['period']))

        # stores the graphs
        simulation.save_plots(graph_plt_path)
        # saves the actual configuration/placement
        simulation_d = simulation.get_simulation_state()
        simulation_d['ga'] = ga.get_optimization_state()
        # save the json
        with open(results_path, 'w') as f:
            json.dump(simulation_d, f)
