#!/usr/bin/env python3

import json
import numpy as np
import random
import yaml
import sys
import time
import math
import os
import traceback
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

import rclpy
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

import gym
from gym import spaces

from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.td3 import TD3
from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_ae import SACAE
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from pic4rl.tasks.Vineyards.pic4rl_environment_camera_depth import Pic4rlEnvironmentCamera
from ament_index_python.packages import get_package_share_directory

from rclpy.executors import SingleThreadedExecutor
from rclpy.executors import ExternalShutdownException


class Pic4rlTraining_Vineyards(Pic4rlEnvironmentCamera):
    def __init__(self):
        super().__init__()
        self.log_check()
        train_params = self.parameters_declaration()

        self.set_parser_list(train_params)
        self.trainer = self.instantiate_agent()

    def instantiate_agent(self):
        """
        ACTION AND OBSERVATION SPACES settings
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

        state =[
        #[0., 15.], # goal_distance 
        #[-math.pi, math.pi], # goal angle or yaw
        ]

        if self.visual_data == 'features':
            for i in range(self.features):
                state = state + [[0., self.max_depth]]
        elif self.visual_data == 'image':
            self.low_state = np.zeros((self.image_height, self.image_width, self.channels),dtype=np.float32)
            self.high_state = self.max_depth*np.ones((self.image_height, self.image_width, self.channels),dtype=np.float32)

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

        self.print_log()

        # OFF-POLICY ALGORITHM TRAINER
        if self.policy_trainer == 'off-policy':
            parser = Trainer.get_argument()
            
            if self.train_policy == 'DDPG':
                self.get_logger().debug('Parsing DDPG parameters...')
                parser = DDPG.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = DDPG(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor = 1e-4,
                    lr_critic = 2e-4,
                    actor_units = (256, 128, 128),
                    critic_units = (256, 128, 128),
                    network='conv',
                    subclassing=False,
                    sigma = 0.2,
                    tau = 0.01,
                    gpu = self.gpu,
                    batch_size= self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon = 1.0, 
                    epsilon_decay = 0.998, 
                    epsilon_min = 0.05,
                    log_level = self.log_level)
                self.get_logger().info('Instantiate DDPG agent...')

            if self.train_policy == 'TD3':
                self.get_logger().debug('Parsing TD3 parameters...')
                parser = TD3.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = TD3(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor = 2e-4,
                    lr_critic = 2e-4,
                    sigma = 0.2,
                    tau = 0.01,
                    epsilon = 1.0, 
                    epsilon_decay = 0.998, 
                    epsilon_min = 0.05,
                    gpu = self.gpu,
                    batch_size= self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    actor_update_freq = 2,
                    policy_noise = 0.2,
                    noise_clip = 0.5,
                    actor_units = (256, 256),
                    critic_units = (256, 256),
                    #network='conv',
                    log_level = self.log_level)
                self.get_logger().info('Instantiate TD3 agent...')
            
            if self.train_policy == 'SAC':
                self.get_logger().debug('Parsing SAC parameters...')
                parser = SAC.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = SAC(
                    state_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action = self.action_space.high,
                    min_action=self.action_space.low,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    #network='conv',
                    tau=5e-3,
                    alpha=.2,
                    auto_alpha=False, 
                    init_temperature=None,
                    gpu = self.gpu,
                    batch_size= self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon = 1.0, 
                    epsilon_decay = 0.996, 
                    epsilon_min = 0.05,
                    log_level = self.log_level)
                self.get_logger().info('Instantiate SAC agent...')

            if self.train_policy == 'SACAE':
                self.get_logger().debug('Parsing SAC-AE parameters...')
                parser = SACAE.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = SACAE(
                    obs_shape = self.observation_space.shape,
                    action_dim = self.action_space.high.size,
                    max_action = self.action_space.high,
                    min_action=self.action_space.low,
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
                    # num_layers_sac=3,
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
                    gpu = self.gpu,
                    batch_size= self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon = 1.0, 
                    epsilon_decay = 0.998, 
                    epsilon_min = 0.05,
                    log_level = self.log_level)
                self.get_logger().info('Instantiate SAC-AE agent...')

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
                    horizon = self.horizon,
                    enable_gae = self.enable_gae,
                    normalize_adv = self.normalize_adv,
                    gpu = self.gpu,
                    batch_size= self.batch_size,
                    log_level = self.log_level)
                self.get_logger().info('Instantiate PPO agent...')

            trainer = OnPolicyTrainer(policy, self, args, test_env=None)
            #self.get_logger().info('Starting process...')
            #trainer()

        return trainer

    def set_parser_list(self, params):
        """
        """
        self.parser_list = []
        for k,v in params.items():
            if v is not None:
                kv = k+'='+str(v)
                self.parser_list.append(kv)
            else:
                self.parser_list.append(k)

        self.parser_list[5] += self.logdir

    def threadFunc(self):
        try:
            self.trainer()
        except Exception:
            self.get_logger().error(f"Error in starting trainer:\n {traceback.format_exc()}")
            return

    def threadFunc_tflite(self):
        while True:
            if self.step_counter == 0:
                observation = self.reset(self.step_counter)
            else:
                observation, reward, done, info = self.step(self.commands)
                self.done = done
            if self.done:
                self.done = False
                self.step_counter = 0
                observation = self.reset(self.step_counter)

            #print(observation[1].shape)
            #print(observation[0].shape)
            self.actor_fp16.set_tensor(self.input_index_state, observation[1])
            self.actor_fp16.set_tensor(self.input_index_image, observation[0])

            self.actor_fp16.invoke()
            self.commands = self.actor_fp16.get_tensor(self.output_index)[0,:]
            #print(self.commands.shape)

            self.step_counter += 1

    def log_check(self):
        """
        Select the ROS2 log level.
        """
        try:
            self.log_level = int(os.environ['LOG_LEVEL'])
        except:
            self.log_level = 20
            self.get_logger().info("LOG_LEVEL not defined, setting default: INFO")

        self.get_logger().set_level(self.log_level)

    def print_log(self):
        """
        """
        for i in range(len(self.log_dict)):
            self.get_logger().info(f"{list(self.log_dict)[i]}: {self.log_dict[list(self.log_dict)[i]]}")

        self.get_logger().info(f"action space shape: {self.action_space.high.size}")
        self.get_logger().info(f"observation space size: {self.observation_space.high.size}\n")

    def parameters_declaration(self):
        """
        """
        main_params_path  = os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'main_params.yaml')
        train_params_path= os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'training_params.yaml')
        
        with open(main_params_path, 'r') as main_params_file:
            main_params = yaml.safe_load(main_params_file)['main_node']['ros__parameters']
        with open(train_params_path, 'r') as train_param_file:
            train_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(namespace='',
        parameters=[
            ('policy', train_params['--policy']),
            ('policy_trainer', train_params['--policy_trainer']),
            ('max_lin_vel', main_params['max_lin_vel']),
            ('min_lin_vel', main_params['min_lin_vel']),
            ('max_ang_vel', main_params['max_ang_vel']),
            ('min_ang_vel', main_params['min_ang_vel']),
            ('gpu', train_params['--gpu']),
            ('batch_size', train_params['--batch-size']),
            ('n_warmup', train_params['--n-warmup'])
            ])

        self.train_policy = self.get_parameter('policy').get_parameter_value().string_value
        self.policy_trainer = self.get_parameter('policy_trainer').get_parameter_value().string_value
        self.min_ang_vel = self.get_parameter('min_ang_vel').get_parameter_value().double_value
        self.min_lin_vel = self.get_parameter('min_lin_vel').get_parameter_value().double_value
        self.max_ang_vel = self.get_parameter('max_ang_vel').get_parameter_value().double_value
        self.max_lin_vel = self.get_parameter('max_lin_vel').get_parameter_value().double_value
        self.gpu = self.get_parameter('gpu').get_parameter_value().integer_value
        self.batch_size = self.get_parameter('batch_size').get_parameter_value().integer_value
        self.n_warmup = self.get_parameter('n_warmup').get_parameter_value().integer_value

        if self.train_policy == 'PPO':
            self.declare_parameters(namespace='',
            parameters=[
                ('horizon', train_params['--horizon']),
                ('normalize_adv', train_params['--normalize-adv']),
                ('enable_gae', train_params['--enable-gae'])
                ])

            self.horizon = self.get_parameter('horizon').get_parameter_value().integer_value
            self.normalize_adv = self.get_parameter('normalize_adv').get_parameter_value().bool_value
            self.enable_gae = self.get_parameter('enable_gae').get_parameter_value().bool_value

        else:
            self.declare_parameters(namespace='',
            parameters=[
                ('memory_capacity', train_params['--memory-capacity'])
                ])

            self.memory_capacity = self.get_parameter('memory_capacity').get_parameter_value().integer_value

        self.log_dict = {
            'policy': train_params['--policy'],
            'max_steps': train_params['--max-steps'],
            'max_episode_steps': train_params['--episode-max-steps'],
            'sensor': main_params['sensor'],
            'visual_data': main_params['visual_data'],
            'features': main_params['features'],
            'gpu': train_params['--gpu']
            }

        return train_params