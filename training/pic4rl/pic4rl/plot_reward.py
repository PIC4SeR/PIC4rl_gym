#!/usr/bin/env python3

# Python libraries
import os
import math
import yaml
import numpy as np
from matplotlib import pyplot as plt

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class Training_Rewards(Node):
    def __init__(self):
        """
        """
        super().__init__('training_rewards')

        train_params_path= os.path.join(
            get_package_share_directory('testing'), 'config', 'training_params.yaml')
        with open(train_params_path, 'r') as train_param_file:
            train_params = yaml.safe_load(train_param_file)['training_params']

        father_path = train_params['--model-dir'] + "/"
        log_path = self.search_log(father_path)
        log_file = open(log_path, 'r')

        train_returns, val_returns = self.load_data(log_file)
        train_mean, val_mean = self.get_cumulated_means(train_returns, val_returns)

        self.get_logger().info("Plotting")
        self.plot_returns(train_returns, val_returns, train_mean, val_mean, father_path)

    def search_log(self, father_path):
        """
        """
        log_path = [f for f in os.listdir(father_path) if f.endswith('.log')]
        if len(log_path) != 1:
            self.get_logger().warn('should be only one log file in the current directory')

        return father_path + log_path[0]

    def load_data(self, log_file):
        """
        """
        train_returns = []
        val_returns = []

        for i, line in enumerate(log_file):
            if line.find('Evaluation')>=0:
                start_of_value = line.find('Average Reward ')+len('Average Reward ')
                end_of_value = line.find(' over')
                val_returns.append(float(line[start_of_value:end_of_value]))
            else:
                start_of_value = line.find('Return: ')+len('Return: ')
                end_of_value = line.find(' Eps:')
                train_returns.append(float(line[start_of_value:end_of_value]))

        return train_returns, val_returns

    def get_cumulated_means(self, train_returns, val_returns):
        """
        """

        train_cumsum    = np.cumsum(train_returns)
        val_cumsum      = np.cumsum(val_returns)
                
        train_mean  = train_cumsum/np.arange(1,len(train_returns)+1)
        val_mean    = val_cumsum/np.arange(1,len(val_returns)+1)

        return train_mean, val_mean

    def plot_returns(self, train_returns, val_returns, train_mean, val_mean, father_path):
        """
        """
        fig,ax = plt.subplots(2, figsize=(20,20))

        ax[0].set_title("Training Results")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward value")
        ax[0].plot(train_returns, label="Train Rewards")
        ax[0].plot(train_mean, label="Train Cumulated Mean")
        ax[0].legend(shadow=True, fancybox=True, loc="lower right")

        ax[1].set_title("Validation Results")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Reward value")
        ax[1].plot(val_returns, label="Validation Rewards")
        ax[1].plot(val_mean, label="Validation Cumulated Mean")
        ax[1].legend(shadow=True, fancybox=True, loc="lower right")

        fig.savefig(father_path + 'rewards_results.png')

        plt.show()


def main(args=None):
    rclpy.init()
    training_rewards = Training_Rewards()
    
    try:
        rclpy.spin(training_rewards)
    except KeyboardInterrupt:
        training_rewards.get_logger().info("Shutting down")

    training_rewards.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
