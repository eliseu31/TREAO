from profiler.visualization import plot_prediction, plot_features_distribution
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import pickle
import json

group_targets = [['ram_mean', 'ram_std'], ['cpu_mean', 'cpu_std'], ['time_mean', 'time_std']]
compact_labels = ['ram', 'cpu', 'time']
MODEL_PATH = '../resources/hyper_model.pkl'
PREDICTIONS_PATH = '../resources/profiling/task_req_prediction.json'


class HyperModel:

    def __init__(self, static_data_path, machine_path, distributions_path):
        # reads the machine data
        with open(machine_path, 'r') as f:
            data_str = f.read()
            machine_specs = json.loads(data_str)['specs']
        ms_df = pd.DataFrame.from_dict(machine_specs, orient='index')
        ms_df.index.name = 'machine_type'
        ms_df.reset_index(inplace=True)
        # reads the function requirements distributions
        with open(distributions_path, 'r') as f:
            data_str = f.read()
            function_req = json.loads(data_str)
        # transforms the data into a dataframe
        df_list = []
        for m_type, m_functions in function_req.items():
            # reads the df
            df = pd.DataFrame.from_dict(m_functions, orient='index')
            df.index.name = 'function'
            df.reset_index(inplace=True)
            # splits each column list
            df[['ram_mean', 'ram_std']] = pd.DataFrame(df['ram'].tolist(), index=df.index)
            df[['cpu_mean', 'cpu_std']] = pd.DataFrame(df['cpu'].tolist(), index=df.index)
            df[['time_mean', 'time_std']] = pd.DataFrame(df['time'].tolist(), index=df.index)
            df.drop(compact_labels, inplace=True, axis=1)
            # formats the indexes
            df['machine_type'] = m_type
            df.set_index(['function', 'machine_type'], inplace=True)
            df_list.append(df)
        # creates the y data
        self.y_data = pd.concat(df_list).sort_index()
        # reads the function static data
        sf_df = pd.read_csv(static_data_path)
        # joins the input data
        sf_df['key'], ms_df['key'] = 1, 1
        self.x_data = pd.merge(sf_df, ms_df, on='key').drop('key', axis=1)
        self.x_data.set_index(['function', 'machine_type'], inplace=True)
        self.x_data.sort_index(inplace=True)
        # print(self.x_data, self.y_data)
        # variable to store the model
        self.pipe = None

    def features_analysis(self):
        # checks the best features for each model
        coefs_list = []
        for model_targets in group_targets:
            linear_regression = LinearRegression()
            linear_regression.fit(self.x_data, self.y_data[model_targets].values)
            print("linear regression train score:", model_targets,
                  linear_regression.score(self.x_data, self.y_data[model_targets]))
            coefs = pd.DataFrame(linear_regression.coef_.transpose(),
                                 columns=model_targets,
                                 index=self.x_data.columns)
            coefs_list.append(coefs)
        # joins the coefs of the 3 models
        coefs_models = pd.concat(coefs_list, axis=1)
        print("linear regression coefficients:\n{0}\n".format(coefs_models))
        # plots the features distributions
        # plot_features_distribution(self.x_data)

    def anova_analysis(self):
        # anova for the machine specs feature
        self._calc_anova(['ram', 'cpu', 'time'], 'machine specs')
        # anova static metrics
        static_metrics = ['bugs', 'calc_length', 'calc_time', 'complexity', 'difficulty', 'effort',
                          'length', 'lloc', 'maintainability', 'ncalls', 'sloc', 'vocabulary', 'volume']
        # anova raw static metrics
        self._calc_anova(['lloc', 'sloc'], 'raw static')
        # anova halstead static metrics
        self._calc_anova(['bugs', 'calc_length', 'calc_time', 'difficulty', 'effort', 'volume'],
                         'halstead metrics')
        # anova halstead static metrics
        self._calc_anova(['calc_length', 'length'], 'halstead length metrics')

    def _calc_anova(self, features_list, print_label):
        column_values = [self.x_data[label] for label in features_list]
        f_value, p_value = stats.f_oneway(*column_values)
        print(print_label, 'features (v-value, p-value):', features_list, f_value, p_value)

    def validate_pipeline(self, features=None, save=False):
        # checks the features
        if not features:
            features = self.x_data.columns
        # uses each function as for test
        function_list = self.x_data.index.levels[0]
        x_index = self.x_data.index.get_level_values('function')
        y_index = self.y_data.index.get_level_values('function')
        # creates te dict to dump the predictions for the optimizer
        f_dict = {f_key: dict() for f_key in self.x_data.index.levels[0]}
        pred_dict = {m_key: f_dict.copy() for m_key in self.x_data.index.levels[1]}
        # init the plots for predictions comparison
        fig = plt.figure(1, figsize=(10, 5))
        subplot_axes = fig.subplots(len(self.x_data.index.levels[1]), len(compact_labels),
                                    sharey=True)
        # experiments each model
        for target_idx, (target_label, target_vars) in enumerate(zip(compact_labels, group_targets)):
            model_predictions = []
            metrics_list = []
            # cross validate using each function
            for test_function in function_list:
                print('validation on:', target_vars, test_function)
                # gets the train and test data
                x_train = self.x_data.loc[x_index != test_function, features]
                x_test = self.x_data.loc[x_index == test_function, features]
                y_train = self.y_data.loc[y_index != test_function, target_vars]
                y_test = self.y_data.loc[y_index == test_function, target_vars]
                # models the regression method
                self.build_pipe(x_train, y_train)
                predictions, metrics = self.predict_task(x_test, y_test)
                metrics_list.append(metrics)
                # stores the predictions on a df
                real_df = pd.DataFrame(y_test.values, columns=['real_mean', 'real_std'], index=y_test.index)
                pred_df = pd.DataFrame(predictions, columns=['predicted_mean', 'predicted_std'], index=y_test.index)
                results_df = pd.concat([real_df, pred_df], axis=1)
                model_predictions.append(results_df)
                # dict dump the predictions
                for m_key, m_dict in pred_dict.items():
                    m_dict[test_function][target_label] = [pred_df.loc[(test_function, m_key), 'predicted_mean'],
                                                           pred_df.loc[(test_function, m_key), 'predicted_std']]
            # print the metric metrics
            mse, r2 = zip(*metrics_list)
            mse = np.absolute(mse)
            r2 = np.absolute(r2)
            print("model test MSE:", np.mean(mse), np.std(mse))
            print("model test r2:", np.mean(r2), np.std(r2))
            # joins the results
            metric_predictions_df = pd.concat(model_predictions)
            # plots the metric results
            self._subplot_metric_predictions(metric_predictions_df, subplot_axes, target_idx)
            subplot_axes[0, target_idx].set_title(target_label.upper())
        # shows the plot
        fig.tight_layout()
        plt.show()
        # stores the predictions
        if save:
            # opens the file and stores the data
            with open(PREDICTIONS_PATH, 'w') as f:
                json.dump(pred_dict, f, indent=2)

    def _subplot_metric_predictions(self, metric_df, subplot_axes, metric_idx):
        # iterates over each machine type
        machine_index = self.y_data.index.get_level_values('machine_type')
        machines_types = metric_df.index.levels[1]
        for m_idx, m_type in enumerate(machines_types):
            # obtains each machine data and plots the data
            m_data = metric_df[machine_index == m_type]
            plot_prediction(m_data, subplot_axes[m_idx, metric_idx])
            # checks if is the first column
            if metric_idx == 0:
                subplot_axes[m_idx, metric_idx].set_ylabel(m_type.upper())

    def predict_task(self, x_test, y_test=None):
        # makes the predictions
        predictions = self.pipe.predict(x_test)
        predictions = np.absolute(predictions)
        print('predictions:', predictions.tolist())
        # checks if receives labels
        metrics = None
        if y_test is not None:
            # obtains the performance metrics
            print('real values', y_test.values.tolist())
            mse = mean_squared_error(y_test, predictions)
            r2 = self.pipe.score(x_test, y_test)
            metrics = (mse, r2)
        # returns the obtained predictions
        return predictions, metrics

    def build_pipe(self, x_train, y_train, save=False):
        # creates the regression model
        model = Ridge(alpha=0.1, random_state=16)
        # builds the pipeline
        self.pipe = Pipeline(steps=[('model', model)])
        self.pipe.fit(x_train, y_train)
        # print("model train r2 score:", self.pipe.score(x_train, y_train))
        # saves the model
        if save:
            # uses pickle to save the model
            with open(MODEL_PATH, 'wb') as file:
                pickle.dump(self.pipe, file)
        # returns the created model
        return self.pipe
