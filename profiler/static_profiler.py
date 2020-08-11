from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import pandas as pd
import numpy as np
import cProfile
import pstats


class ProfilerHandler:

    def __init__(self, modules_dict):
        # dict which stores the module profiler
        self.static_profiler = dict()
        # iterates over each module to process
        for mod_key, mod_data in modules_dict.items():
            # creates the profiler
            self.static_profiler[mod_key] = StaticProfiler(mod_data[0])
            self.static_profiler[mod_key].cyclomatic_complexity()
            self.static_profiler[mod_key].ast_analysis()
            self.static_profiler[mod_key].lines_analysis()
            self.static_profiler[mod_key].maintainability()
            self.static_profiler[mod_key].function_calls(mod_data[1], [mod_data[2]])

    def save_metrics(self, path):
        values_matrix = []
        row_indexes = []
        column_names = None
        # iterates over each module profiler
        for mod_key, profiler in self.static_profiler.items():
            # gets the data
            profiler_t = sorted(profiler.metrics.items(), key=lambda x: x[0])
            metric_names, values = zip(*profiler_t)
            values_matrix.append(list(values))
            row_indexes.append(mod_key)
            if not column_names:
                column_names = list(metric_names)
        # creates the df
        df = pd.DataFrame(values_matrix, index=row_indexes, columns=column_names)
        df.index.name = 'function'
        print(df)
        df.to_csv(path)


class StaticProfiler:

    def __init__(self, module_path):
        # prints some info
        print('profiling the module at {0}'.format(module_path))
        # reads the file
        with open(module_path) as f:
            self.module_code = f.read()
        # store the obtained results
        self.metrics = dict()

    def cyclomatic_complexity(self):
        # parses the data
        visitor = ComplexityVisitor.from_code(self.module_code)
        # computes the mean complexity
        complexity_list = [f.complexity for f in visitor.functions]
        self.metrics['complexity'] = np.mean(complexity_list)
        print('cyclomatic complexity: {0}'.format(self.metrics['complexity']))

    def lines_analysis(self):
        # parses the data
        result = analyze(self.module_code)
        # stores the interesting metrics
        self.metrics['lloc'] = result.lloc
        self.metrics['sloc'] = result.sloc
        print('lloc: {0}, sloc: {1}'.format(self.metrics['lloc'], self.metrics['sloc']))

    def ast_analysis(self):
        # parses the data
        ast_root = h_visit(self.module_code)
        # stores the metrics
        self.metrics['vocabulary'] = ast_root.total.vocabulary
        self.metrics['length'] = ast_root.total.length
        self.metrics['calc_length'] = ast_root.total.calculated_length
        self.metrics['volume'] = ast_root.total.volume
        self.metrics['difficulty'] = ast_root.total.difficulty
        self.metrics['effort'] = ast_root.total.effort
        self.metrics['calc_time'] = ast_root.total.time
        self.metrics['bugs'] = ast_root.total.effort
        print('vocabulary: {0}, length: {1}, calculated length: {2}'.format(self.metrics['vocabulary'],
                                                                            self.metrics['length'],
                                                                            self.metrics['calc_length']))
        print('volume: {0}, difficulty: {1}, effort: {2}'.format(self.metrics['volume'],
                                                                 self.metrics['difficulty'],
                                                                 self.metrics['effort']))
        print('time: {0}, bugs: {1}'.format(self.metrics['calc_time'], self.metrics['bugs']))

    def maintainability(self):
        # parses the code
        result = mi_visit(self.module_code, multi=True)
        # stores the obtained result
        self.metrics['maintainability'] = result
        print('maintainability: {0}'.format(self.metrics['maintainability']))

    def function_calls(self, method, method_args, verbose=False):
        # profiles the method
        prof = cProfile.Profile()
        prof.enable()
        method(*method_args)
        prof.disable()
        # formats the obtained
        p_stats = pstats.Stats()
        p_stats.add(prof)
        self.metrics['ncalls'] = p_stats.total_calls
        print('number of calls: {0}'.format(self.metrics['ncalls']))
        # prints the detailed info
        if verbose:
            p_stats.sort_stats('ncalls')
            p_stats.print_stats()
