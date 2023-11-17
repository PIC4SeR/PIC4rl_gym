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

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
from pic4rl.tasks.Following.pic4rl_environment_lidar_pf import (
    Pic4rlEnvironment_Lidar_PF,
)
from ament_index_python.packages import get_package_share_directory

from rclpy.executors import SingleThreadedExecutor
from rclpy.executors import ExternalShutdownException


class Pic4rlLidar_PF(Pic4rlEnvironment_Lidar_PF):
    def __init__(self, get_entity_client):
        super().__init__(get_entity_client)
        self.log_check()
        train_params = self.parameters_declaration()

        if self.tflite_flag:
            self.actor_fp16 = tf.lite.Interpreter(
                model_path="~/inference/actor_fp16.tflite"
            )
            self.actor_fp16.allocate_tensors()
            self.input_index_image = self.actor_fp16.get_input_details()[0]["index"]
            self.input_index_state = self.actor_fp16.get_input_details()[1]["index"]
            self.output_index = self.actor_fp16.get_output_details()[0]["index"]
            self.commands = [0.0, 0.0]
            self.step_counter = 0
            self.done = False

        else:
            self.set_parser_list(train_params)
            self.trainer = self.instantiate_agent()

    def instantiate_agent(self):
        """
        ACTION AND OBSERVATION SPACES settings
        """
        action = [self.min_ang_vel, self.max_ang_vel]  # w_speed

        low_action = [action[0]]
        high_action = [action[1]]

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(1,), dtype=np.float32
        )

        state = [
            [0.0, 15.0],  # goal_distance
            [-math.pi, math.pi],  # goal angle or yaw
            [self.min_ang_vel, self.max_ang_vel],
        ]

        if len(state) > 0:
            low_state = []
            high_state = []
            for i in range(len(state)):
                low_state.append(state[i][0])
                high_state.append(state[i][1])

            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        # Set Epsilon-greedy starting value for exploration policy (minimum 0.05)
        epsilon = 0.6

        self.print_log()
        if self.mode == "testing":
            self.epsilon = 0.0
        else:
            self.epsilon = epsilon

        self.print_log()

        # OFF-POLICY ALGORITHM TRAINER
        if self.policy_trainer == "off-policy":
            parser = Trainer.get_argument()
            if self.train_policy == "DDPG":
                self.get_logger().debug("Parsing DDPG parameters...")
                parser = DDPG.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = DDPG(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor=2e-4,
                    lr_critic=2e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    subclassing=False,
                    sigma=0.2,
                    tau=0.01,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon=self.epsilon,
                    epsilon_decay=0.998,
                    epsilon_min=0.05,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate DDPG agent...")

            if self.train_policy == "TD3":
                self.get_logger().debug("Parsing TD3 parameters...")
                parser = TD3.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = TD3(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr_actor=2e-4,
                    lr_critic=2e-4,
                    sigma=0.2,
                    tau=0.01,
                    epsilon=self.epsilon,
                    epsilon_decay=0.998,
                    epsilon_min=0.05,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    actor_update_freq=2,
                    policy_noise=0.2,
                    noise_clip=0.5,
                    actor_units=(256, 128, 128),
                    critic_units=(256, 128, 128),
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate TD3 agent...")

            if self.train_policy == "SAC":
                self.get_logger().debug("Parsing SAC parameters...")
                parser = SAC.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = SAC(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    max_action=self.action_space.high,
                    min_action=self.action_space.low,
                    lr=2e-4,
                    lr_alpha=3e-4,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    tau=5e-3,
                    alpha=0.2,
                    auto_alpha=False,
                    init_temperature=None,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    n_warmup=self.n_warmup,
                    memory_capacity=self.memory_capacity,
                    epsilon=self.epsilon,
                    epsilon_decay=0.996,
                    epsilon_min=0.05,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate SAC agent...")

            trainer = Trainer(policy, self, args, test_env=None)
            # self.get_logger().info('Starting process...')
            # trainer()

        # ON-POLICY ALGORITHM TRAINER
        if self.policy_trainer == "on-policy":
            parser = OnPolicyTrainer.get_argument()

            if self.train_policy == "PPO":
                self.get_logger().debug("Parsing PPO parameters...")
                parser = PPO.get_argument(parser)
                args = parser.parse_args(self.parser_list)
                policy = PPO(
                    state_shape=self.observation_space.shape,
                    action_dim=self.action_space.high.size,
                    is_discrete=False,
                    max_action=self.action_space.high,
                    lr_actor=1e-3,
                    lr_critic=3e-3,
                    actor_units=(256, 256),
                    critic_units=(256, 256),
                    hidden_activation_actor="relu",
                    hidden_activation_critic="relu",
                    clip=True,
                    clip_ratio=0.2,
                    horizon=self.horizon,
                    enable_gae=self.enable_gae,
                    normalize_adv=self.normalize_adv,
                    gpu=self.gpu,
                    batch_size=self.batch_size,
                    log_level=self.log_level,
                )
                self.get_logger().info("Instanciate PPO agent...")

            trainer = OnPolicyTrainer(policy, self, args, test_env=None)
            # self.get_logger().info('Starting process...')
            # trainer()

        return trainer

    def set_parser_list(self, params):
        """ """
        self.parser_list = []
        for k, v in params.items():
            if v is not None:
                kv = k + "=" + str(v)
                self.parser_list.append(kv)
            else:
                self.parser_list.append(k)

        self.parser_list[5] += self.logdir

    def threadFunc(self):
        try:
            self.trainer()
        except Exception:
            self.get_logger().error(
                f"Error in starting trainer:\n {traceback.format_exc()}"
            )
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

            self.actor_fp16.set_tensor(self.input_index_state, observation[1])
            self.actor_fp16.set_tensor(self.input_index_image, observation[0])

            self.actor_fp16.invoke()
            self.commands = self.actor_fp16.get_tensor(self.output_index)[0, :]

            self.step_counter += 1

    def log_check(self):
        """
        Select the ROS2 log level.
        """
        try:
            self.log_level = int(os.environ["LOG_LEVEL"])
        except:
            self.log_level = 20
            self.get_logger().info("LOG_LEVEL not defined, setting default: INFO")

        self.get_logger().set_level(self.log_level)

    def print_log(self):
        """ """
        for i in range(len(self.log_dict)):
            self.get_logger().info(
                f"{list(self.log_dict)[i]}: {self.log_dict[list(self.log_dict)[i]]}"
            )

        self.get_logger().info(f"action space shape: {self.action_space.high.size}")
        self.get_logger().info(
            f"observation space size: {self.observation_space.high.size}\n"
        )

    def parameters_declaration(self):
        """ """
        self.declare_parameter("package_name", "pic4rl")
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        main_params_path = os.path.join(
            get_package_share_directory(self.package_name), "config", "main_params.yaml"
        )
        train_params_path = os.path.join(
            get_package_share_directory(self.package_name),
            "config",
            "training_params.yaml",
        )

        with open(main_params_path, "r") as main_params_file:
            main_params = yaml.safe_load(main_params_file)["main_node"][
                "ros__parameters"
            ]
        with open(train_params_path, "r") as train_param_file:
            train_params = yaml.safe_load(train_param_file)["training_params"]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("policy", train_params["--policy"]),
                ("policy_trainer", train_params["--policy_trainer"]),
                ("max_lin_vel", main_params["max_lin_vel"]),
                ("min_lin_vel", main_params["min_lin_vel"]),
                ("max_ang_vel", main_params["max_ang_vel"]),
                ("min_ang_vel", main_params["min_ang_vel"]),
                ("tflite_flag", train_params["--tflite_flag"]),
                ("tflite_model_path", train_params["--tflite_model_path"]),
                ("gpu", train_params["--gpu"]),
                ("batch_size", train_params["--batch-size"]),
                ("n_warmup", train_params["--n-warmup"]),
            ],
        )

        self.train_policy = (
            self.get_parameter("policy").get_parameter_value().string_value
        )
        self.policy_trainer = (
            self.get_parameter("policy_trainer").get_parameter_value().string_value
        )
        self.min_ang_vel = (
            self.get_parameter("min_ang_vel").get_parameter_value().double_value
        )
        self.min_lin_vel = (
            self.get_parameter("min_lin_vel").get_parameter_value().double_value
        )
        self.max_ang_vel = (
            self.get_parameter("max_ang_vel").get_parameter_value().double_value
        )
        self.max_lin_vel = (
            self.get_parameter("max_lin_vel").get_parameter_value().double_value
        )
        self.tflite_flag = (
            self.get_parameter("tflite_flag").get_parameter_value().bool_value
        )
        self.tflite_model_path = (
            self.get_parameter("tflite_model_path").get_parameter_value().string_value
        )
        self.gpu = self.get_parameter("gpu").get_parameter_value().integer_value
        self.batch_size = (
            self.get_parameter("batch_size").get_parameter_value().integer_value
        )
        self.n_warmup = (
            self.get_parameter("n_warmup").get_parameter_value().integer_value
        )

        if self.train_policy == "PPO":
            self.declare_parameters(
                namespace="",
                parameters=[
                    ("horizon", train_params["--horizon"]),
                    ("normalize_adv", train_params["--normalize-adv"]),
                    ("enable_gae", train_params["--enable-gae"]),
                ],
            )

            self.horizon = (
                self.get_parameter("horizon").get_parameter_value().integer_value
            )
            self.normalize_adv = (
                self.get_parameter("normalize_adv").get_parameter_value().bool_value
            )
            self.enable_gae = (
                self.get_parameter("enable_gae").get_parameter_value().bool_value
            )

        else:
            self.declare_parameters(
                namespace="",
                parameters=[("memory_capacity", train_params["--memory-capacity"])],
            )

            self.memory_capacity = (
                self.get_parameter("memory_capacity")
                .get_parameter_value()
                .integer_value
            )

        self.log_dict = {
            "policy": train_params["--policy"],
            "max_steps": train_params["--max-steps"],
            "max_episode_steps": train_params["--episode-max-steps"],
            "sensor": main_params["sensor"],
            "gpu": train_params["--gpu"],
        }

        return train_params
