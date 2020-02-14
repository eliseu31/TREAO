from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import time
import os
import sys
import psutil

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from profiler.sorting_algorithms import bubble_sort
from profiler.sorting_algorithms import selection_sort
from profiler.sorting_algorithms import insertion_sort
from profiler.sorting_algorithms import merge_sort
from profiler.sorting_algorithms import quick_sort
from profiler.sorting_algorithms import heap_sort


def process_algorithm(method, record_size, measures_list):
    # get the time before the execution
    start_time = time.time()

    # measure the time for this algorithm
    method(record_size)

    # get the resulting delta time
    delta_time = round(time.time() - start_time, 3)
    # measure the ram
    ram_results = round(psutil.Process(os.getpid()).memory_info().rss / 1000000, 3)
    # append the results
    measures_list.put(ram_results)
    measures_list.put(delta_time)


if __name__ == "__main__":
    machine_name = sys.argv[1]
    trial_idx = sys.argv[2]
    # list of algorithms to call
    algorithms = {'bubble_sort': bubble_sort,
                  'selection_sort': selection_sort,
                  'insertion_sort': insertion_sort,
                  'merge_sort': merge_sort,
                  'quick_sort': quick_sort,
                  'heap_sort': heap_sort}
    results = []

    print('\nTRIAL #', trial_idx)
    # initialize a array with the size record
    record2sort = np.random.rand(2048).tolist()
    process_metrics = Queue()

    # sort the list using each algorithm
    for name, algorithm in algorithms.items():
        # initialize a process
        process = Process(target=process_algorithm, args=[algorithm,
                                                          record2sort.copy(),
                                                          process_metrics])
        process.start()
        # measure the cpu while runs the process
        process2measure = psutil.Process(process.pid)
        cpu_results = []
        while process2measure.is_running():
            cpu_value = process2measure.cpu_percent()
            cpu_results.append(cpu_value)
            try:
                process2measure.wait(0.01)
            except psutil.TimeoutExpired:
                pass

        # waits until the end
        process.join()

        cpu_value = round(float(np.mean(cpu_results)), 2)
        # append the results to the array
        result_sample = [trial_idx, machine_name, name, cpu_value]
        while not process_metrics.empty():
            result_sample.append(process_metrics.get())
        results.append(result_sample)

        print(name, ':', result_sample[4], result_sample[5], cpu_value)

    # save the results in a csv
    path2append = os.path.join(os.path.dirname(sys.path[0]),
                               'data',
                               'profiling_results.csv')
    file = open(path2append, 'a')
    with file:
        for row in results:
            row_str = '\n' + ','.join(str(item) for item in row)
            file.write(row_str)

    # store the specs of the machine
    process_specs = psutil.Process(os.getpid())
