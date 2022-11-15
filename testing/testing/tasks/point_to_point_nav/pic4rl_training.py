#!/usr/bin/env python3

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step

import json
import numpy as np
import random
import sys
import time
import math

import gym
from gym import spaces


from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.td3 import TD3
from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_ae import SACAE
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from pic4rl.pic4rl_environment import Pic4rlEnvironment
from pic4rl.pic4rl_environment_lidar import Pic4rlEnvironmentLidar
from pic4rl.pic4rl_environment_camera_depth import Pic4rlEnvironmentCamera

from rclpy.executors import SingleThreadedExecutor
from rclpy.executors import ExternalShutdownException

import threading

import yaml

class Pic4rlTraining(Pic4rlEnvironmentCamera):
    def __init__(self):
        super().__init__()
        rclpy.logging.set_logger_level('pic4rl_training', 10)
        self.declare_parameters(namespace='',
        parameters=[
            ('policy', 'SAC'),
            ('policy_trainer', 'off-policy'),
            ('trainer_params', '/root/gym_ws/src/PIC4rl_gym/training/pic4rl/config/training_params.yaml'),
            ('max_lin_vel', 0.5),
            ('min_lin_vel', 0.0),
            ('max_ang_vel', 1.0),
            ('min_ang_vel', -1.0),
            ('sensor', 'camera'),
            ('lidar_points', 36),
            ('visual_data', 'features'),
            ('features', 12544),
            ('channels', 1),
            ('image_width', 112),
            ('image_height', 112)
            ])

        qos = QoSProfile(depth=10)

        self.train_policy = self.get_parameter('policy').get_parameter_value().string_value
        self.policy_trainer = self.get_parameter('policy_trainer').get_parameter_value().string_value
        self.training_params = self.get_parameter('trainer_params').get_parameter_value().string_value
        self.sensor = self.get_parameter('sensor').get_parameter_value().string_value
        self.min_ang_vel = self.get_parameter('min_ang_vel').get_parameter_value().double_value
        self.min_lin_vel = self.get_parameter('min_lin_vel').get_parameter_value().double_value
        self.max_ang_vel = self.get_parameter('max_ang_vel').get_parameter_value().double_value
        self.max_lin_vel = self.get_parameter('max_lin_vel').get_parameter_value().double_value
        self.lidar_points = self.get_parameter('lidar_points').get_parameter_value().integer_value
        self.visual_data = self.get_parameter('visual_data').get_parameter_value().string_value
        self.features = self.get_parameter('features').get_parameter_value().integer_value
        self.channels = self.get_parameter('channels').get_parameter_value().integer_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value

        self.set_parser_list()
        self.trainer = self.instantiate_agent()

        #self.pic4_environment = Pic4rlEnvironmentLidar()

    def instantiate_agent(self):
        """
        ACTION AND OBSERVATION SPACES settings
        """
        """
        actions
        """
        action =[
        [self.min_lin_vel, self.max_lin_vel], # x_speed 
        #[self.min_lin_vel, self.max_lin_vel], # y_speed
        [self.min_ang_vel, self.max_ang_vel] # w_speed
        ]

        low_action = []
        high_action = []
        for i in range(len(action)):
            low_action.append(action[i][0])
            high_action.append(action[i][1])

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            shape=(2,),
            dtype=np.float32
        )
        self.get_logger().info('action space shape: {}'.format(self.action_space.high.size))
        """
        state: TODO change state according to the task
        """
        state =[
        [0., 15.], # goal_distance 
        [-math.pi, math.pi] # goal angle or yaw
        ]
        """
        sensor observation: TODO change state according to the sensor parameters
        """
        if self.sensor == 'lidar':
            for i in range(self.lidar_points):
                state = state + [[0., 12.]]
        if self.sensor == 'camera':
            if self.visual_data == 'features':
                for i in range(self.features):
                    state = state + [[0., 5.]]
            elif self.visual_data == 'image':
                self.low_state = np.zeros((self.image_height, self.image_width, self.channels),dtype=np.float32)
                self.high_state = 5.*np.ones((self.image_height, self.image_width, self.channels),dtype=np.float32)

        if len(state)>0:
            low_state = []
            high_state = []
            for i in range(len(state)):
                low_state.append(state[i][0])
                high_state.append(state[i][1])

            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.get_logger().info('observation space size: {}'.format(self.observation_space.shape))
        # OFF-POLICY ALGORITHM TRAINER
        if self.policy_trainer == 'off-policy':
            parser = Trainer.get_argument()
            
            if self.train_policy == 'DDPG':
                self.get_logger().debug('Parsing DDPG parameters...')
                parser = DDPG.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = DDPG(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    lr_actor = 3e-4,
                    lr_critic = 3e-4,
                    actor_units = (256, 256),
                    critic_units = (256, 256),
                    sigma = 0.2,
                    tau = 0.01,

                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05)
                self.get_logger().info('Instanciate DDPG agent...')

            if self.train_policy == 'TD3':
                self.get_logger().debug('Parsing TD3 parameters...')
                parser = TD3.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = TD3(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    lr_actor = 3e-4,
                    lr_critic = 3e-4,
                    sigma = 0.2,
                    tau = 0.01,
                    epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    actor_update_freq = 2,
                    policy_noise = 0.2,
                    noise_clip = 0.5,
                    critic_units = (256, 256))
                self.get_logger().info('Instanciate TD3 agent...')
            
            if self.train_policy == 'SAC':
                self.get_logger().debug('Parsing SAC parameters...')
                parser = SAC.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = SAC(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action = self.action_space.high,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    num_layers_sac=2,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    tau=5e-3,
                    alpha=.2,
                    auto_alpha=False, 
                    init_temperature=None,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    epsilon = 0.8, epsilon_decay = 0.992, epsilon_min = 0.05)
                self.get_logger().info('Instanciate SAC agent...')

            if self.train_policy == 'SACAE':
                self.get_logger().debug('Parsing SAC-AE parameters...')
                parser = SACAE.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = SACAE(
                    obs_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action = self.action_space.high,
                    n_conv_layers=4,
                    n_conv_filters=32,
                    feature_dim=50,
                    tau_encoder=0.05,
                    tau_critic=0.01,
                    auto_alpha=False,
                    alpha=.2,
                    lr_sac=1e-3,
                    lr_encoder=1e-3,
                    lr_decoder=1e-3,
                    num_layers_sac=3,
                    actor_units=(256, 128, 128),
                    critic_units=(256, 128, 128),
                    update_critic_target_freq=2,
                    update_actor_freq=2,
                    lr_alpha=1e-4,
                    init_temperature=0.1,
                    stop_q_grad=False,
                    lambda_latent_val=1e-06,
                    decoder_weight_lambda=1e-07,
                    skip_making_decoder=False,
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"],
                    n_warmup=self.param_dict["training_params"]["--n-warmup"],
                    memory_capacity=self.param_dict["training_params"]["--memory-capacity"],
                    epsilon = 0.1, epsilon_decay = 0.998, epsilon_min = 0.05)
                self.get_logger().info('Instanciate SAC-AE agent...')

            trainer = Trainer(policy, self, args, test_env=None)
            #self.get_logger().info('Starting process...')
            #trainer()

        # ON-POLICY ALGORITHM TRAINER
        if self.policy_trainer == 'on-policy':
            parser = OnPolicyTrainer.get_argument()
            
            if self.train_policy == 'PPO':
                self.get_logger().debug('Parsing PPO parameters...')
                parser = PPO.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                print(args)
                policy = PPO(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    is_discrete = False,
                    max_action=self.action_space.high,
                    lr_actor = 1e-3,
                    lr_critic = 3e-3,
                    actor_units = (256, 256),
                    critic_units = (256, 256),
                    hidden_activation_actor="relu",
                    hidden_activation_critic="relu",
                    clip = True,
                    clip_ratio = 0.2,
                    horizon = self.param_dict["training_params"]["--horizon"],
                    enable_gae = self.param_dict["training_params"]["--enable-gae"],
                    normalize_adv = self.param_dict["training_params"]["--normalize-adv"],
                    gpu = self.param_dict["training_params"]["--gpu"],
                    batch_size= self.param_dict["training_params"]["--batch-size"])
                self.get_logger().info('Instanciate PPO agent...')

            trainer = OnPolicyTrainer(policy, self, args, test_env=None)
            #self.get_logger().info('Starting process...')
            #trainer()

        return trainer
    def set_parser_list(self):
        with open(self.training_params, 'r') as f:
            self.param_dict = yaml.load(f)

        self.parser_list = []
        for k,v in self.param_dict['training_params'].items():
            if v is not None:
                kv = k+'='+str(v)
                self.parser_list.append(kv)
            else:
                self.parser_list.append(k)

    def threadFunc(self):
        try:
            self.trainer()
        except:
            return

def main(args=None):
    rclpy.init()
    pic4rl_training= Pic4rlTraining()
    th = threading.Thread(target=pic4rl_training.threadFunc)
    th.start()
    try:
        rclpy.spin(pic4rl_training)
    except:
        pic4rl_training.destroy_node()
        th.join()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
