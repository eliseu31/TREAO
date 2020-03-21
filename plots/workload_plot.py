import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

if __name__ == '__main__':
    # root path
    root_path = os.path.join(os.path.dirname(sys.path[0]))
    # read the data
    data_path = os.path.join(root_path, 'resources', 'workload_results.csv')
    df = pd.read_csv(data_path)

    # init the figure
    fig = plt.figure(0, figsize=(4, 4.5))
    # plot the bar plot
    plt.bar(df.index/2, df['fitness'], color='skyblue', width=0.2)
    plt.ylabel('Fitness')
    plt.ylim([0, 0.7])
    plt.xlabel('Machines/Tasks')
    # plot the line plot
    line_ax = plt.twinx()
    line_ax.plot((df.index/2).values, df['generations'].values, color='darkblue', marker='.')
    line_ax.set_ylabel('Generations')

    x_labels = df['machines'].map(str) + '/' + df['tasks'].map(str)
    plt.xticks(df.index/2, x_labels)
    plt.title('Workload Evolution')
    plt.tight_layout()
    plt.show()
