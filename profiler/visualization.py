from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(model_results, axes):
    # gets the function names
    x_pos = np.arange(len(model_results))
    # plots the real values
    axes.barh(x_pos-0.2, model_results['real_mean'], xerr=model_results['real_std'],
              color='g', height=0.4, label='real value')
    # plots the predicted values
    axes.barh(x_pos+0.2, model_results['predicted_mean'], xerr=model_results['predicted_std'],
              color='b', height=0.4, label='prediction')
    x_labels = model_results.index.levels[0]
    axes.set_yticks(x_pos)
    axes.set_yticklabels(x_labels.tolist())
    axes.legend(loc=1)


def plot_features_distribution(df):
    # plots the histogram of variables
    machine_types = df.index.levels[1]
    machine_index = df.index.get_level_values('machine_type')
    for m_type in machine_types:
        # obtains each machine data and plots the data
        m_data = df[machine_index == m_type]
        scatter_matrix(m_data)
        plt.show()
