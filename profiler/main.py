from resources.methods.bubble_sort import bubble_sort
from resources.methods.insertion_sort import insertion_sort
from resources.methods.selection_sort import selection_sort
from resources.methods.quick_sort import quick_sort
from resources.methods.merge_sort import merge_sort
from resources.methods.heap_sort import heap_sort
from profiler.static_profiler import ProfilerHandler
from profiler.hyper_model import HyperModel
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

# method arguments
record = np.random.rand(1024).tolist()
modules_dict = {'bubble_sort': ['../resources/methods/bubble_sort.py', bubble_sort, record.copy()],
                'insertion_sort': ['../resources/methods/insertion_sort.py', insertion_sort, record.copy()],
                'selection_sort': ['../resources/methods/selection_sort.py', selection_sort, record.copy()],
                'quick_sort': ['../resources/methods/quick_sort.py', quick_sort, record.copy()],
                'merge_sort': ['../resources/methods/merge_sort.py', merge_sort, record.copy()],
                'heap_sort': ['../resources/methods/heap_sort.py', heap_sort, record.copy()]}
METRICS_PATH = '../resources/profiling/static_metrics.csv'
MACHINES_PATH = '../resources/machine_specs/machines_specs.json'
FUNCTIONS_PATH = '../resources/profiling/task_req.json'

if __name__ == '__main__':
    # starts the static profiling
    # handler = ProfilerHandler(modules_dict)
    # handler.save_metrics(METRICS_PATH)
    # model
    hm = HyperModel(METRICS_PATH, MACHINES_PATH, FUNCTIONS_PATH)
    hm.features_analysis()
    # hm.anova_analysis()
    hm.validate_pipeline(['complexity', 'effort', 'ncalls', 'ram', 'cpu'], save=True)
    # hm.evaluate_model()
