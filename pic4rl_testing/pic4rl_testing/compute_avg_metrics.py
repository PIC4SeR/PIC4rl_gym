#!/usr/bin/env python3

# Python libraries
import os
import math
import yaml
import numpy as np
from matplotlib import pyplot as plt
from ament_index_python.packages import get_package_share_directory


class Compute_Metrics():
    def __init__(self):
        """
        """
        father_path = '/root/agri_ws/src/Results/vineyard_curve_results/Segmentation_tree_20230416_154520.212714_jackal_ugv_vineyard_curve/'
        log_path = self.search_yaml(father_path)

        self.metrics_results = {"Clearance_time":None, 
                "Cumulative_heading_average":None, 
                "Mean_velocities":None,
                "Std_velocities":None, 
                "row_crop_path_MAE":None,
                "row_crop_path_MSE":None}

        self.num_episodes = 1
        self.num_run = 3

        self.data_loaded = self.load_data(log_path)
        #train_mean, val_mean = self.get_cumulated_means(train_returns, val_returns)
        self.result_file_path = father_path+'/metrics_average.txt'
        self.copy_metrics()

    def search_yaml(self, father_path):
        """
        """
        yaml_path = [f for f in os.listdir(father_path) if f.endswith('.yaml')]
        if len(yaml_path) != 1:
            self.get_logger().warn('should be only one yaml file in the current directory')

        return father_path + yaml_path[0]

    def load_data(self, log_file):
        """
        """
        with open(log_file, "r") as stream:
            try:
                data_loaded = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        return data_loaded

    def copy_metrics(self,):
        """
        """
        num_exp = self.num_episodes*self.num_run
        for i in range(num_exp):
            #ep_number = int(str(i)[-1])+1
            ep = 'episode'+str(i+1)
            results = self.data_loaded[ep]

            results['Mean_velocities'] = results['Mean_velocities'][0]
            results['Std_velocities'] = results['Std_velocities'][1]

            for m in results.keys():
                if m in self.metrics_results.keys():
                    self.metrics_results[m] = results[m]

            self.store_metrics(ep, self.metrics_results)

    def get_cumulated_means(self, metric_results):
        """
        """
        metric_cumsum    = np.cumsum(metric_results)
        metric_mean  = train_cumsum/np.arange(1,len(metric_results)+1)

        return metric_mean

    def store_metrics(self, exp_tag, metrics_results):
        """
        """
        # add extension if it does not have it
        if not self.result_file_path.endswith(".txt"):
            self.result_file_path += '.txt'

        file_was_created = os.path.exists(self.result_file_path)

        # open the file
        file = open(self.result_file_path,'a+')
        if(file is None):
            print("RESULT METRICS FILE NOT CREATED! FILE: %s" % self.result_file_path)

        # if the file is new, create a header
        if file_was_created == False:
            #file.write('experiment_tag')
            #file.write('\t')
            for m in metrics_results.keys():
                file.write(m)
                file.write('\t')
            file.write('\n')
        
        # write the data
        #file.write(exp_tag)
        #file.write('\t')
        for v in metrics_results.values():
            file.write(str(v))
            file.write('\t')
        file.write('\n')
        file.close()


def main(args=None):
    compute_metrics = Compute_Metrics()

if __name__ == '__main__':
    main()