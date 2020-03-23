from simulation.machine_data import MachinesManager
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import random
import json
import warnings

warnings.filterwarnings("ignore")


class GraphManager:

    def __init__(self, graph_path, machines_path, backup_path, df_path=None):
        # starts the machine manager
        self.mm = MachinesManager(backup_path, df_path)

        # read the machines data
        with open(machines_path, 'r') as machines_file:
            machines_data = json.loads(machines_file.read())
        # receive the machines present in the network
        self.available_machines = machines_data['machines']
        self.machines_specs = machines_data['specs']
        # receive all the network delay between machines
        network_tuples = [(tuple(sorted(item_dict['edge'])), item_dict['delay'])
                          for item_dict in machines_data['network_delay']]
        self.network_delays = dict(network_tuples)

        # read the graph data
        with open(graph_path, 'r') as graph_file:
            graph_data = json.loads(graph_file.read())
        # convert the graph nodes
        graph_nodes = [(int(k), v) for k, v in graph_data['nodes'].items()]
        # associate a random machine to each pipeline node
        for node_idx, node_data in graph_nodes:
            # stores the machine id in the graph
            node_data['machine_id'] = random.choice(list(self.available_machines.keys()))
        # graph creation
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(graph_nodes)
        self.graph.add_edges_from(graph_data['edges'])
        # update the network edges
        self.update_edges()

        # initializes all the results from cost functions
        self.utility_values = {'ram': 0, 'cpu': 0, 'l_path': 0, 'n_traffic': 0, 's_mac': 0, 'clusters': 0}
        self.longest_path = []
        # norm values
        self.norm = {'ram': 0.02, 'cpu': 0.004, 'l_path': 0.1, 'n_traffic': 0.45, 's_mac': 0.15, 'clusters': 0.2}

        # filter each color
        colors_list = ['green', 'blue', 'yellow', 'dimgray', 'red', 'darkviolet', 'orange']
        self.color_machines = dict(zip(self.available_machines.keys(), colors_list))

        # figure to plot the simulation
        self.fig_simulation = None
        self.axes = dict()
        self.points = {'ram': [], 'cpu': [], 'l_path': [], 'n_traffic': [], 's_mac': [], 'clusters': []}

    def simulate_graph(self):
        # predict the next pipeline
        self.predict_pipeline()
        # calculate the utility functions
        self.search_longest_path()
        self.calc_avg_cost()
        self.calc_machines_cost()
        self.calc_cluster_cost()

    def update_combination(self, machines_sequence):
        # iterate over the graph
        for node_idx, machine_id in enumerate(machines_sequence):
            # update the value of the machine
            self.graph.nodes[node_idx]['machine_id'] = machine_id

    def get_actual_combination(self):
        # get the nodes_id and machines
        nodes_comb = self.graph.nodes(data='machine_id')
        return nodes_comb

    def read_utility_values(self):
        # joins all the values in one dict
        utility_values = {'l_path': self.utility_values['l_path'] * self.norm['l_path'],
                          'ram': self.utility_values['ram'] * self.norm['ram'],
                          'cpu': self.utility_values['cpu'] * self.norm['cpu'],
                          'n_traffic': self.utility_values['n_traffic'] * self.norm['n_traffic'],
                          's_mac': self.utility_values['s_mac'] * self.norm['s_mac'],
                          'clusters': self.utility_values['clusters'] * self.norm['clusters']}
        return utility_values

    def predict_pipeline(self):
        # iterate over the graph
        for node_idx, node_data in self.graph.nodes(data=True):
            machine_id = node_data['machine_id']
            f_name = node_data['function']
            # map the machine id to a type
            machine_type = self.available_machines[machine_id]['machine_type']
            # get the predictions from the machine
            predictions_tuple = self.mm.machines_data[machine_type].predict_resources(f_name)
            node_data['time'], node_data['ram'], node_data['cpu'] = predictions_tuple
        # update the network data
        self.update_edges()

    def update_edges(self):
        large_edges, short_edges = [], []
        # iterate over the edges
        for edge_source, edge_target in self.graph.edges:
            # from the node idx get the machine id
            source_machine_id = self.graph.nodes[edge_source]['machine_id']
            target_machine_id = self.graph.nodes[edge_target]['machine_id']
            # using the machine id tuple get the network delay
            edge_tuple = tuple(sorted((source_machine_id, target_machine_id)))
            if edge_tuple in self.network_delays:
                # if exists sets the value
                edge_delay = self.network_delays[edge_tuple]
                # appends to the large edges
                large_edges.append((edge_source, edge_target))
            else:
                # if not exists set the delay to 0
                edge_delay = 0
                # appends to the large edges
                short_edges.append((edge_source, edge_target))
            # updates the edge delay in the graph
            self.graph[edge_source][edge_target]['network_delay'] = edge_delay
        # return the edges
        return large_edges, short_edges

    def find_parallel_paths(self):
        # get the start nodes
        node_in_degree = dict(self.graph.in_degree)
        min_value = min(node_in_degree.values())
        min_pairs = filter(lambda x: x[1] == min_value, node_in_degree.items())
        in_nodes, _ = zip(*min_pairs)
        # get the end nodes
        node_out_degree = dict(self.graph.out_degree)
        min_value = min(node_out_degree.values())
        min_pairs = filter(lambda x: x[1] == min_value, node_out_degree.items())
        out_nodes, _ = zip(*min_pairs)
        # using combinations get all the possible paths
        point_combinations = itertools.product(in_nodes, out_nodes)
        # return a list with the paths
        return list(point_combinations)

    def find_clusters(self):
        # group nodes according to the machines
        group_nodes = itertools.groupby(sorted(self.graph.nodes(data='machine_id'), key=lambda x: x[1]),
                                        key=lambda x: x[1])
        # clusters output
        n_clusters = dict(zip(self.available_machines.keys(), [0]*len(self.available_machines.keys())))
        # for each machine check the number of clusters
        for machine_id, group in group_nodes:
            # gets the nodes list
            nodes, _ = zip(*group)
            # creates the sub-graph based in the machine nodes
            subgraph = self.graph.subgraph(nodes)
            # get the number of connected components
            n_clusters[machine_id] = nx.number_connected_components(subgraph.to_undirected())
        # returns the number of clusters
        return n_clusters

    def search_longest_path(self):
        # resets the variable
        self.longest_path = []
        self.utility_values['l_path'] = 0
        max_cost = 0
        # get the points to find the longest path
        points = self.find_parallel_paths()
        # iterate over each possible source and target
        for source, target in points:
            # get the possible paths between each other
            possible_paths = nx.all_simple_paths(self.graph, source, target)
            # iterate over each path
            for path in possible_paths:
                # get the path cost
                path_cost = self.calc_longest_path(path)
                # check if is the bigger
                if path_cost > max_cost:
                    # calculates the medium value
                    cost_avg = path_cost / len(path)
                    # updates the maximum value
                    max_cost = path_cost
                    # replace the maximum
                    self.longest_path = path
                    self.utility_values['l_path'] = cost_avg

    def calc_longest_path(self, path):
        # gets the costs of each node
        nodes_cost_values = map(lambda x: self.graph.nodes[x]['time'], path)
        nodes_cost = sum(nodes_cost_values)
        # get the links cost
        edges_cost_values = map(lambda x: self.graph[x[0]][x[1]]['network_delay'],
                                zip(path[:], path[1:]))
        edges_cost = sum(edges_cost_values)
        # return the sum of both
        return nodes_cost + edges_cost

    def calc_cluster_cost(self):
        # gets the number of clusters
        clusters = self.find_clusters()
        # calculates the diff between 1 cluster per machine
        self.utility_values['clusters'] = sum(map(lambda x: abs(x[1] - 1), clusters.items()))
        # returns the cost value
        return self.utility_values['clusters']

    def calc_avg_cost(self):
        # get all the values
        values = [(node_data['ram'], node_data['cpu']) for _, node_data in self.graph.nodes(data=True)]
        ram_values, cpu_values = zip(*values)
        # get the network values
        net_values = [self.graph[source][target]['network_delay'] for source, target in self.graph.edges]
        # update the value
        self.utility_values['ram'] = np.mean(ram_values) / len(self.available_machines)
        self.utility_values['cpu'] = np.mean(cpu_values) / len(self.available_machines)
        self.utility_values['n_traffic'] = np.mean(net_values)
        # return the value
        return self.utility_values['ram'], self.utility_values['cpu'], self.utility_values['n_traffic']

    def calc_machines_cost(self):
        # iterates over the machines
        for mac_id, mac_d in self.available_machines.items():
            # machine type
            mac_r = self.machines_specs[mac_d['machine_type']]
            # filters the nodes that are associated to that machine
            mac_costs = [
                (lambda x: x['time'] / mac_r['time'] + x['ram'] / mac_r['ram'] + x['cpu'] / mac_r['cpu'])(node_data)
                for _, node_data in self.graph.nodes(data=True)
                if node_data['machine_id'] == mac_id]
            # saves the costs
            mac_d['costs'] = sum(mac_costs)
        # calculates the difference between costs
        costs_list = [abs(mac_1['costs'] - mac_2['costs'])
                      for mac_1, mac_2 in itertools.combinations(self.available_machines.values(), 2)]
        self.utility_values['s_mac'] = sum(costs_list)
        # return the costs
        return self.utility_values['s_mac']

    def print_graph(self):
        # iterate over the graph
        for node_idx, node_data in self.graph.nodes(data=True):
            # prints each node data
            print(node_idx, node_data)

    def init_simulation(self):
        plt.ion()
        # create the figure
        self.fig_simulation = plt.figure(2, figsize=(10, 5))
        # create a grid in the figure
        grid = plt.GridSpec(3, 4, figure=self.fig_simulation)
        # subplot for the graph
        self.axes['graph'] = plt.subplot(grid[:, 1:])
        # subplot for the longest path
        self.axes['utility_values'] = plt.subplot(grid[:2, 0])
        # subplot for machines costs
        self.axes['mac'] = plt.subplot(grid[2, 0])

    def flush_plot(self, refresh=0):
        # while the figure is open simulate the graph
        if plt.fignum_exists(2):
            # plot the graph
            self.plot_graph()
            # plot the utility functions
            self.plot_utility_functions()
            # draw the canvas
            self.fig_simulation.canvas.draw()
            self.fig_simulation.canvas.flush_events()
            # sleeps
            plt.tight_layout()
            plt.pause(refresh)
            # if plots return True
            return True
        else:
            return False

    def plot_utility_functions(self):
        # plot the utility graph
        self.axes['utility_values'].clear()
        self.axes['utility_values'].set_title('Utility Functions')
        # gets the normalized values
        norm_values = self.read_utility_values()
        # plot each line
        for line_label, line_point in self.points.items():
            # append the actual value
            line_point.append(norm_values[line_label])
            self.axes['utility_values'].plot(line_point, label=line_label)
        # plots the legend
        self.axes['utility_values'].legend()
        # plot machine costs
        self.axes['mac'].clear()
        self.axes['mac'].set_title('Machines Costs')
        # get the graph variables
        mac_var = [(mac_id, mac_data['costs'], self.color_machines[mac_id])
                   for mac_id, mac_data in self.available_machines.items()]
        mac_id, mac_costs, mac_color = zip(*mac_var)
        self.axes['mac'].bar(mac_id, mac_costs, 0.2, color=mac_color)

    def plot_graph(self):
        # clear the actual graph
        self.axes['graph'].clear()
        self.axes['graph'].set_title('Graph Structure')
        pos = nx.spectral_layout(self.graph)

        # set the nodes size
        node_size = {'low': 500, 'normal': 1000, 'overload': 2000}
        # find each cpu level node
        cpu_nodes = dict()
        cpu_nodes['overload'] = [node_id for (node_id, cpu) in self.graph.nodes(data='cpu') if cpu >= 90]
        cpu_nodes['normal'] = [node_id for (node_id, cpu) in self.graph.nodes(data='cpu') if 90 > cpu >= 50]
        cpu_nodes['low'] = [node_id for (node_id, cpu) in self.graph.nodes(data='cpu') if cpu < 50]
        # plot the nodes
        for node_id, machine_id in self.graph.nodes(data='machine_id'):
            # get the cpu class
            cpu_class, _ = list(filter(lambda x: node_id in x[1], cpu_nodes.items()))[0]
            # draw the node
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[node_id], node_size=node_size[cpu_class],
                                   node_color=self.color_machines[machine_id], ax=self.axes['graph'])
        # node labels
        nx.draw_networkx_labels(self.graph, pos, ax=self.axes['graph'])

        # finds the large and short edges
        large_edges, short_edges = self.update_edges()
        # get the edges of the longest path
        edges_long_path = list(zip(self.longest_path[:], self.longest_path[1:]))
        # group in 4 different groups (including the longest path)
        edges_labels = dict()
        edges_labels['red_short'] = list(filter(lambda x: x in edges_long_path, short_edges))
        edges_labels['black_short'] = list(filter(lambda x: x not in edges_long_path, short_edges))
        edges_labels['red_large'] = list(filter(lambda x: x in edges_long_path, large_edges))
        edges_labels['black_large'] = list(filter(lambda x: x not in edges_long_path, large_edges))
        # plot the edges
        for edge_source, edge_target in self.graph.edges:
            # get the cpu class
            cpu_class, _ = list(filter(lambda x: edge_target in x[1], cpu_nodes.items()))[0]
            # check if is in the longest path
            edge_color = 'red' if (edge_source, edge_target) in edges_long_path else 'black'

            edge_line = nx.draw_networkx_edges(self.graph, pos, edgelist=[(edge_source, edge_target)],
                                               edge_color=edge_color, node_size=node_size[cpu_class],
                                               width=2, ax=self.axes['graph'])
            # dashed if is between 2 machines
            if (edge_source, edge_target) in large_edges and edge_line is not None:
                edge_line[0].set_linestyle('dashed')

        # add labels associating each node to one machine
        machines_pairs = self.graph.nodes(data='machine_id')
        # get the machine ids
        _, machine_ids = zip(*machines_pairs)
        machines_association = []
        for machine_id in set(machine_ids):
            # generates a list of pairs (node_id, machine_id)
            nodes_pair = filter(lambda x: x[1] == machine_id, machines_pairs)
            # unzip the tuples
            nodes_list, _ = zip(*nodes_pair)
            # append the pair (machine_id, nodes_list) to the list
            machines_association.append((machine_id, nodes_list))
        # generate the string
        str_list = map(lambda x: '{0}: {1}'.format(x[0], list(x[1])), machines_association)
        str2plot = '\n'.join(str_list)
        # add the annotation to the graph
        at = AnchoredText(str2plot, loc='upper right')
        self.axes['graph'].add_artist(at)

    def save_plots(self, path):
        # saves the figure
        self.fig_simulation.savefig(path)

    def get_simulation_state(self):
        # create the output dict
        output_dict = dict()
        # convert to json the graph
        output_dict['nodes'] = dict(self.graph.nodes(data=True))
        output_dict['edges'] = [[[source, target], data] for source, target, data in self.graph.edges(data=True)]
        # saves the utility values
        output_dict['costs'] = dict((v_name, [value, value*self.norm[v_name]])
                                    for v_name, value in self.utility_values.items())
        # returns the data
        return output_dict
