import os
import cv2
import math
import yaml
import reverb
import shutil
import logging
import numpy as np
from numpy import savetxt
import subprocess
import random
import sys
import time
import datetime
from pathlib import Path

import tensorflow as tf

from tf_agents.networks import sequential
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.dqn import dqn_agent

from ament_index_python.packages import get_package_share_directory

# DQN Agent

class EnvUtils():
    """
    """
    def get_ros_params(self, pkg, path, data):
        """
        """
        class Parser():
            def __init__(self, params, path):
                self.policy          = params['policy']
                self.sensor          = params['sensor']
                self.logdir          = params['logdir']
                self.ep_max_steps    = int(float(params['ep_max_steps']))
                self.change_ep       = params['change_ep']
                self.starting_eps    = params['starting_eps']
                self.robot_name      = params['robot_name']
                self.world_name      = params['world_name']
                self.data_name       = params['data_name']
                self.max_lin_vel     = params['max_lin_vel']
                self.min_lin_vel     = params['min_lin_vel']
                self.max_ang_vel     = params['max_ang_vel']
                self.min_ang_vel     = params['min_ang_vel']
                self.robot_type      = params['robot_type']
                self.robot_radius    = params['robot_radius']
                self.robot_size      = params['robot_size']
                self.coll_tol        = params['collision_tolerance']
                self.warn_tol        = params['warning_tolerance']
                self.goal_tolerance  = params['goal_tolerance']             
                self.lidar_distance  = params['laser_param']['max_distance']
                self.lidar_points    = params['laser_param']['num_points']
                self.imu_enabled     = params['imu_enabled']
                self.camera_enabled  = params['camera_enabled']
                self.lidar_enabled   = params['lidar_enabled']
                self.rgb_topic       = params['sensors_topic']['rgb_topic']
                self.depth_topic     = params['sensors_topic']['depth_topic']
                self.laser_topic     = params['sensors_topic']['laser_topic']
                self.imu_topic       = params['sensors_topic']['imu_topic']
                self.odom_topic      = params['sensors_topic']['odom_topic']
                self.rgb_width       = params['rgb_param']['width']
                self.rgb_height      = params['rgb_param']['height']
                self.show_rgb        = params['rgb_param']['show_image']
                self.dist_cutoff     = params['depth_param']['dist_cutoff']
                self.depth_width     = params['depth_param']['width']
                self.depth_height    = params['depth_param']['height']
                self.show_depth      = params['depth_param']['show_image']
                self.lidar_enables   = params['lidar_enabled']
                self.visual_data     = params['visual_data']
                self.features        = params['features']

                self.retrain         = 'retrain' in params.keys()
                self.evaluate        = 'evaluate' in params.keys()

                self.params_path     = path

        total_path = os.path.join(
            get_package_share_directory(pkg), path)

        with open(total_path, 'r') as file:
            env_params = yaml.safe_load(file)[data]
            
        parser = Parser(env_params, total_path)
        
        return parser

    def get_data(self, pkg, path, data):
        """
        """
        total_path = os.path.join(
            get_package_share_directory(pkg), path)

        with open(total_path, 'r') as file:
            params = yaml.safe_load(file)[data]

        return params

    def create_logdir(self, policy, sensor, logdir):
        """
        """
        _logdir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{sensor}_{policy}/"
        Path(os.path.join(logdir, _logdir)).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(logdir, _logdir, 'screen_logger.log'), 
            level=logging.INFO)
        return _logdir

    def process_odom(self, goal_pose, odom):
        """
        """
        goal_distance = math.sqrt(
            (goal_pose[0]-odom[0])**2
            + (goal_pose[1]-odom[1])**2)

        path_theta = math.atan2(
            goal_pose[1]-odom[1],
            goal_pose[0]-odom[0])

        goal_angle = path_theta - odom[2]

        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        goal_info = [goal_distance, goal_angle]
        robot_pose = [odom[0], odom[1], odom[2]]

        return goal_info, robot_pose
 
    def get_random_goal(self, episode, initial_pose):
        """
        """
        if episode < 6 or episode % 25 == 0:
            x = 0.55
            y = 0.55
        else:
            x = random.randrange(-29, 29) / 10.0
            y = random.randrange(-29, 29) / 10.0

        x += initial_pose[0]
        y += initial_pose[1]

        return [x, y]

    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def normalize(self, v):
        norm=np.linalg.norm(v)
        if norm==0:
            return v, norm
            #norm=np.finfo(v.dtype).eps
        return v/norm, norm

    def normalize_angle(self, theta):
        # theta should be in the range [-pi, pi]
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi
        return theta

    def quat_to_euler(self, qz, qw):
        """
        """
        t1 = 2*(qw*qz)
        t2 = 1 - 2*(qz*qz)

        Wz = math.atan2(t1,t2)

        return Wz

    def euler_to_quat(self, Wz):
        """
        """
        qz = np.sin(Wz/2)
        qw = np.cos(Wz/2)

        return qz, qw

    def tf_compose(self, robot_pose, goal_pose):
        """
        This method composes two consecutive reference frames.

        For example, it can return the pose of the goal, converted from 
        fixed reference frame to robot reference frame, with robot_pose and 
        goal_pose given in fixed reference frame.
        """
        xr, yr, zqr, wqr = tuple(robot_pose)
        zr = quat_to_euler(zqr, wqr)

        A = np.zeros([4,4], dtype=float)
        A[0,0] = math.cos(zr)
        A[1,0] = math.sin(zr)
        A[0,1] = -A[1,0]
        A[1,1] = A[0,0]
        A[0,3] = xr
        A[1,3] = yr
        A[2,3] = 0
        A[3,3] = 1
        A[2,2] = 1
        A = np.linalg.inv(A)
        
        return np.matmul(A, goal_pose).tolist()
        
    def tf_decompose2(self, robot_pose, goal_pose):
        """
        This method decomposes two consecutive reference frames.

        For example, it can return the pose of the goal, converted from 
        robot reference frame to fixed reference frame, with robot_pose given in 
        fixed reference frame and goal_pose given in robot reference frame.
        """ 
        xr, yr, zqr, wqr = tuple(robot_pose)
        zr = quat_to_euler(zqr, wqr)

        A = np.zeros([4,4], dtype=float)
        A[0,0] = math.cos(zr)
        A[1,0] = math.sin(zr)
        A[0,1] = -A[1,0]
        A[1,1] = A[0,0]
        A[0,3] = xr
        A[1,3] = yr
        A[2,3] = 0
        A[3,3] = 1
        A[2,2] = 1
        
        return np.matmul(A, goal_pose).tolist()

    def tf_decompose(self, robot_pose, goal_pose):
        """
        This method decomposes two consecutive reference frames.

        For example, it can return the pose of the goal, converted from 
        robot reference frame to fixed reference frame, with robot_pose given in 
        fixed reference frame and goal_pose given in robot reference frame.
        """ 
        xr, yr, zr = tuple(robot_pose)

        A = np.zeros([4,4], dtype=float)
        A[0,0] = math.cos(zr)
        A[1,0] = math.sin(zr)
        A[0,1] = -A[1,0]
        A[1,1] = A[0,0]
        A[0,3] = xr
        A[1,3] = yr
        A[2,3] = 0
        A[3,3] = 1
        A[2,2] = 1
        
        return np.matmul(A, goal_pose).tolist()

class TrainUtils():
    """
    """
    def get_dense_layer(self, num_units):
        """
        """
        dense_layer = tf.keras.layers.Dense(
            num_units,
            activation  = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.VarianceScaling(
                scale = 2.0, 
                mode = 'fan_in', 
                distribution = 'truncated_normal'
                )
            )

        return dense_layer

    # Discrete agents
    def get_q_layer(self, num_actions):
        """
        """
        q_layer = tf.keras.layers.Dense(
            num_actions,
            activation = None,
            kernel_initializer = tf.keras.initializers.RandomUniform(
                minval = -0.03, 
                maxval = 0.03
                ),
            bias_initializer = tf.keras.initializers.Constant(-0.2)
            )

        return q_layer

    def get_dqn_agent(self,params,observation_spec,action_spec,time_step_spec):
        """
        """
        num_actions = action_spec.maximum-action_spec.minimum+1

        dense_layers = [self.get_dense_layer(num_units) for num_units in params.layer_units]
        q_net = sequential.Sequential(dense_layers + [self.get_q_layer(num_actions)])
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0)
            )

        agent.initialize()

        return agent

    # Actor-Critic continuous agents
    def get_critic_net(self, observation_spec, action_spec, layer_units):
        """
        """
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=tuple(layer_units),
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform'
            )

        return critic_net

    def get_actor_net(self, observation_spec, action_spec, layer_units):
        """
        """
        actor_net = actor_network.ActorNetwork(
            observation_spec, 
            action_spec,
            fc_layer_params=tuple(layer_units)
            )

        return actor_net

    def get_distributed_actor_net(self, observation_spec, action_spec, layer_units):
        """
        """
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec, 
            action_spec,
            fc_layer_params=tuple(layer_units),
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork)
            )

        return actor_net

    def get_ddpg_agent(self,params,observation_spec,action_spec,time_step_spec):
        """
        """
        critic_net = self.get_critic_net(
            observation_spec, action_spec, params.layer_units)
        actor_net = self.get_actor_net(
            observation_spec, action_spec, params.layer_units)

        agent = ddpg_agent.DdpgAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=params.learning_rate),
            critic_optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=params.learning_rate),
            target_update_tau=params.tau,
            target_update_period=params.update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=params.gamma,
            reward_scale_factor=params.rew_scale_factor,
            train_step_counter=tf.Variable(0)
            )

        agent.initialize()

        return agent

    def get_td3_agent(self,params,observation_spec,action_spec,time_step_spec):
        """
        """
        critic_net = self.get_critic_net(
            observation_spec, action_spec, params.layer_units)
        actor_net = self.get_actor_net(
            observation_spec, action_spec, params.layer_units)     

        agent = td3_agent.Td3Agent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=params.learning_rate),
            critic_optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=params.learning_rate),
            target_update_tau=params.tau,
            target_update_period=params.update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=params.gamma,
            reward_scale_factor=params.rew_scale_factor,
            train_step_counter=tf.Variable(0)
            )

        agent.initialize()

        return agent

    def get_sac_agent(self,params,observation_spec,action_spec,time_step_spec):
        """
        """
        critic_net = self.get_critic_net(
            observation_spec, action_spec, params.layer_units)
        actor_net = self.get_distributed_actor_net(
            observation_spec, action_spec, params.layer_units)

        agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=params.learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=params.learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=params.alpha_learn_rate),
            target_update_tau=params.tau,
            target_update_period=params.update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=params.gamma,
            reward_scale_factor=params.rew_scale_factor,
            train_step_counter=tf.Variable(0)
            )

        agent.initialize()

        return agent

    # Metrics

    def compute_avg_return(self, environment, policy, num_episodes=10):
        """
        """
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
            steps = 0

            while not time_step.is_last():
                steps += 1
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
    #             print(action_step)
    #             print(time_step)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes

        return avg_return.numpy()[0]

    # Reply Buffer

    def create_replay_buffer(self, agent, replay_buffer_max_length=int(1e5)):
        """
        """
        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size     = replay_buffer_max_length,
            sampler      = reverb.selectors.Uniform(),
            remover      = reverb.selectors.Fifo(),
            rate_limiter = reverb.rate_limiters.MinSize(1),
            signature    = replay_buffer_signature
        )

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            agent.collect_data_spec,
            table_name      = table_name,
            sequence_length = 2,
            local_server    = reverb_server
        )

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length = 2
        )
        
        dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=128,
        num_steps=2).prefetch(3)

        iterator = iter(dataset)
        # print(iterator)
        
        return replay_buffer, rb_observer, iterator

    # Save and load checkpoints and policy

    def save_ckpt(self, path, agent, replay_buffer, policy=True):
        """
        """
        checkpointer = common.Checkpointer(
            ckpt_dir=path,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter
        )
        checkpointer.save(agent.train_step_counter)
        
        if policy:
            _policy_saver = policy_saver.PolicySaver(agent.policy)
            _policy_saver.save(path)

    def load_ckpt(self, path, agent, replay_buffer, policy=False):
        """
        """
        checkpointer = common.Checkpointer(
            ckpt_dir=path,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter
        )
        checkpointer.initialize_or_restore()
            
    # Parse parameters

    def get_parameters(self, path):
        """
        """
        class Parser():
            def __init__(self, params, path):
                self.policy             = params['policy']
                self.sensor             = params['sensor']
                self.logdir             = params['logdir']
                self.max_steps          = int(float(params['max_steps']))
                self.ep_max_steps       = int(float(params['ep_max_steps']))
                self.eval_interval      = int(float(params['eval_interval']))
                self.eval_episodes      = int(float(params['eval_episodes']))
                self.layer_units        = params['layer_units']
                self.learning_rate      = float(params['learning_rate'])
                self.rb_size            = int(float(params['rb_size']))
                self.rb_steps_per_iter  = params['rb_steps_per_iter']
                self.warmup_steps       = int(float(params['warmup_steps']))
                self.task               = params['task']
                self.model_dir          = params['model_dir']
                self.alpha_learn_rate   = float(params['alpha_learn_rate'])
                self.tau                = float(params['tau'])
                self.update_period      = params['update_period']
                self.gamma              = float(params['gamma'])
                self.rew_scale_factor   = params['rew_scale_factor']
                self.retrain            = 'retrain' in params.keys()
                self.evaluate           = 'evaluate' in params.keys()

                self.params_path        = path
                
                if self.retrain and self.evaluate:
                    raise ValueError("You can either retrain or evaluate!")

        total_path = os.path.join(
            get_package_share_directory('pic4rl'), path)   

        with open(total_path, 'r') as train_param_file:
            params = yaml.safe_load(train_param_file)['params']
            
        parser = Parser(params, total_path)
        
        return parser
            
    # Logging

    def set_logger(self, path=None):
        """
        """    
        logFormatter = logging.Formatter("%(message)s")
        logging.root.setLevel(logging.INFO)
        rootLogger = logging.getLogger()

        if path:
            fileHandler = logging.FileHandler(path)
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        return rootLogger

    # Path definition

    def define_path(self, params):
        """
        """
        tm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
        name = f"{tm}_{params.sensor}_{params.policy}"
        
        results_path    = f"{params.logdir}/{name}"
        log_path        = f"{results_path}/{name}.log"

        if not os.path.isdir(results_path):
            os.makedirs(results_path)

        shutil.copy(params.params_path, f"{results_path}/params.yaml")
            
        return results_path, log_path