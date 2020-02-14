from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


class MachinesManager:

    def __init__(self, backup_path, df_path=None):
        # parse the data set according to the machines
        self.machines_data = {}

        # fits the distribution
        if df_path:
            # dict to save
            fit_results = dict()
            # read the dataframe from the data path
            df = pd.read_csv(df_path)
            # unique list of machines
            machines_types = df['machine_type'].unique()
            # iterate over the machine ids
            for machine_type in machines_types:
                # creates the machine obj
                self.machines_data[machine_type] = MachineData(machine_type, df=df)
                # fits to each machine
                self.machines_data[machine_type].fit()
                # stores the results
                fit_results[machine_type] = self.machines_data[machine_type].distributions
            # saves the file
            with open(backup_path, 'w') as backup_file:
                json.dump(fit_results, backup_file)
        else:
            # loads the distributions
            with open(backup_path, 'r') as backup_file:
                backup = backup_file.read()
                distributions = json.loads(backup)
            # maps the distributions
            for machine_type, distribution in distributions.items():
                # creates the machine obj
                self.machines_data[machine_type] = MachineData(machine_type, distributions=distribution)


class MachineData:

    def __init__(self, machine_type, df=None, distributions=None):
        # names of the algorithms
        self.algorithm_names = ['bubble_sort', 'heap_sort', 'insertion_sort',
                                'merge_sort', 'quick_sort', 'selection_sort']
        # fit the distributions
        self.distributions = distributions if distributions else dict()
        # saves the actual df
        self.df = df
        # stores the type
        self.my_type = machine_type

    def fit(self):
        # select each machine data
        m_df = self.df.loc[self.df['machine_type'] == self.my_type]
        # iterate over all the algorithms
        for name_idx, f_name in enumerate(self.algorithm_names):
            f_values = m_df.loc[m_df['function'] == f_name]
            # time metric
            time_param, _ = self.fit_distribution(f_values['time'].values)
            # ram metric
            ram_param, _ = self.fit_distribution(f_values['ram'].values)
            # cpu usage metric
            cpu_param, _ = self.fit_distribution(f_values['cpu'].values)
            # add the values to the dict
            self.distributions[f_name] = {'time': time_param, 'ram': ram_param, 'cpu': cpu_param}

    @staticmethod
    def fit_distribution(values):
        # use the normal distribution
        norm_param = stats.norm.fit(values)
        # use multimodal distribution
        kde = stats.gaussian_kde(values, bw_method=0.5)
        # return the distributions
        return norm_param, kde

    def predict_resources(self, f_name):
        # time predictions
        time_prediction = np.random.normal(self.distributions[f_name]['time'][0],
                                           self.distributions[f_name]['time'][1])
        # ram predictions
        ram_prediction = np.random.normal(self.distributions[f_name]['ram'][0],
                                          self.distributions[f_name]['ram'][1])
        # cpu predictions
        cpu_prediction = np.random.normal(self.distributions[f_name]['cpu'][0],
                                          self.distributions[f_name]['cpu'][1])
        # return the 3 values
        return time_prediction, ram_prediction, cpu_prediction

    def plot_histogram(self):
        # one subplot to each function measure
        fig = plt.figure(1)
        axes = fig.subplots(nrows=3, ncols=6)

        for name_idx, f_name in enumerate(self.algorithm_names):
            # time metric
            self.plot_distribution(f_name, 'time', axes[0, name_idx], 30)
            # ram metric
            self.plot_distribution(f_name, 'ram', axes[1, name_idx], 30)
            # cpu usage metric
            self.plot_distribution(f_name, 'cpu', axes[2, name_idx], 30)
            # general graphic
            axes[0, name_idx].set_title(f_name)
            axes[0, 0].set_ylabel('Time')
            axes[1, 0].set_ylabel('RAM Usage')
            axes[2, 0].set_ylabel('CPU Usage')

        # plot the graphic
        plt.show()

    def plot_distribution(self, f_name, metric, ax, bins):
        # select each machine data
        m_df = self.df.loc[self.df['machine_type'] == self.my_type]
        # get the metric for that function
        values = m_df.loc[m_df['function'] == f_name][metric].values
        # plot the histogram
        histogram, bins = np.histogram(values, bins=bins)
        ax.hist(values, bins=bins, density=True, color="skyblue")
        # use the normal distribution
        norm_dist = stats.norm.pdf(bins,
                                   loc=self.distributions[f_name][metric][0],
                                   scale=self.distributions[f_name][metric][1])
        ax.plot(bins, norm_dist, label='gaussian')
        # use multimodal distribution
        kde = stats.gaussian_kde(values, bw_method=0.5)
        ax.plot(bins, kde.pdf(bins), label='multimodal')
        # make the legend
        ax.set_yticks([])
        ax.legend()
