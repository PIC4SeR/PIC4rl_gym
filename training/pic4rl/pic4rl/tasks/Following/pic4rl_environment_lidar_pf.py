#!/usr/bin/env python3

import os
import numpy as np
from numpy import savetxt
import math
import subprocess
import json
import random
import sys
import time
import datetime
import yaml
import logging
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetEntityState
from pic4rl.utils.get_entity_client import GetEntityClient


class Pic4rlEnvironment_Lidar_PF(Node):
    def __init__(self, get_entity_client):
        """
        """
        super().__init__('pic4rl_training_lidar')
        goals_path      = os.path.join(
            get_package_share_directory('pic4rl'), 'goals_and_poses')
        main_params_path  = os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'main_params.yaml')
        train_params_path= os.path.join(
            get_package_share_directory('pic4rl'), 'config', 'training_params.yaml')
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), 
            'models/goal_box/model.sdf'
            )
        
        with open(main_params_path, 'r') as main_params_file:
            main_params = yaml.safe_load(main_params_file)['main_node']['ros__parameters']
        with open(train_params_path, 'r') as train_param_file:
            train_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(
            namespace   = '',
            parameters  = [
                ('data_path', main_params['data_path']),
                ('change_goal_and_pose', train_params['--change_goal_and_pose']),
                ('starting_episodes', train_params['--starting_episodes']),
                ('timeout_steps', train_params['--episode-max-steps']),
                ('robot_name', main_params['robot_name']),
                ('goal_tolerance', main_params['goal_tolerance']),
                ('lidar_dist', main_params['laser_param']['max_distance']),
                ('lidar_points', main_params['laser_param']['num_points']),
                ('update_frequency', main_params['update_frequency'])
                ]
            )

        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path      = os.path.join(goals_path, self.data_path)
        self.change_episode = self.get_parameter(
            'change_goal_and_pose').get_parameter_value().integer_value
        self.starting_episodes = self.get_parameter(
            'starting_episodes').get_parameter_value().integer_value
        self.timeout_steps  = self.get_parameter(
            'timeout_steps').get_parameter_value().integer_value
        self.robot_name     = self.get_parameter(
            'robot_name').get_parameter_value().string_value
        self.goal_tolerance = self.get_parameter(
            'goal_tolerance').get_parameter_value().double_value
        self.lidar_distance = self.get_parameter(
            'lidar_dist').get_parameter_value().double_value
        self.lidar_points   = self.get_parameter(
            'lidar_points').get_parameter_value().integer_value
        self.params_update_freq   = self.get_parameter(
            'update_frequency').get_parameter_value().double_value

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        create_logdir(train_params['--policy'], main_params['sensor'], train_params['--logdir'])
        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'follow_vel',
            qos)

        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)

        self.reset_world_client     = self.create_client(
            Empty, 'reset_world')
        self.pause_physics_client   = self.create_client(
            Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(
            Empty, 'unpause_physics')

        self.episode_step       = 0
        self.previous_twist     = Twist()
        self.episode            = 0
        self.collision_count    = 0
        self.t0                 = 0.0

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]

        self.get_entity_client = get_entity_client

        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """
        """
        twist = Twist()
        twist.angular.z = float(action)
        self.episode_step = episode_step

        observation, reward, done = self._step(twist)
        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step = False):
        """
        """
        self.get_logger().debug("sending action...")
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        lidar_measurements, goal_info, robot_pose, collision = self.get_sensor_data()

        self.get_logger().debug("checking events...")
        done, event = self.check_events(lidar_measurements, goal_info, robot_pose, collision)

        if not reset_step:
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(twist, lidar_measurements, goal_info, robot_pose, done, event)

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(twist, lidar_measurements, goal_info, robot_pose)
        else:
            reward = None
            observation = None

        self.update_state(twist,lidar_measurements, goal_info, robot_pose, done, event)

        return observation, reward, done

    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path,'r'))

        return data["initial_pose"], data["goals"], data["poses"]

    def spin_sensors_callbacks(self):
        """
        """
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        
    def send_action(self,twist):
        """
        """
        #self.get_logger().debug("unpausing...")
        #self.unpause()

        #self.get_logger().debug("publishing twist...")
        self.cmd_vel_pub.publish(twist)

        # Regulate frequency of send action if needed
        # freq, t1 = compute_frequency(self.t0)
        # t0 = t1
        # frequency_control(self.params_update_freq)

        #self.get_logger().debug("pausing...")
        #self.pause()

    def get_sensor_data(self):
        """
        """
        sensor_data = {}
        sensor_data["scan"], collision = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()
        
        if sensor_data["scan"] is None:
            sensor_data["scan"] = (np.ones(self.lidar_points)*self.lidar_distance).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0,0.0,0.0]

        goal_info, robot_pose = process_odom(sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]

        return lidar_measurements, goal_info, robot_pose, collision

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """
        """
        if collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        if goal_info[0] < self.goal_tolerance:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            return True, "goal"

        if self.episode_step+1 >= self.timeout_steps:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout")
            return True, "timeout"

        return False, "None"

    def get_reward(self,twist,lidar_measurements, goal_info, robot_pose, done, event):
        """
        """
        yaw_reward = (1-2*math.sqrt(math.fabs(goal_info[1]/math.pi)))
        smooth_reward = -math.fabs(self.previous_twist.angular.z-twist.angular.z)

        reward = yaw_reward + smooth_reward

        if event == "goal":
            reward += 1000
        elif event == "collision":
            reward += -200
        self.get_logger().debug(str(reward))

        return reward

    def get_observation(self, twist,lidar_measurements, goal_info, robot_pose):
        """
        """
        state_list = goal_info
        state_list.append(twist.angular.z)

        state = np.array(state_list,dtype = np.float32)

        return state

    def update_state(self,twist,lidar_measurements, goal_info, robot_pose, done, event):
        """
        """
        self.previous_twist = twist
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

    def reset(self, n_episode, tot_steps, evaluate=False):
        """
        """
        self.episode = n_episode
        self.evaluate = evaluate
        logging.info(f"Total_episodes: {'evaluate' if evaluate else n_episode}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n")
        print()
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        _,_,_, = self._step(reset_step = True)
        observation,_,_, = self._step()

        return observation
    
    def new_episode(self):
        """
        """
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()

        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service not available, waiting again...')
        self.reset_world_client.call_async(req)
        
        if self.episode % self.change_episode == 0. or self.evaluate:
            self.index = int(np.random.uniform()*len(self.poses)) -1 

        self.get_logger().debug("Respawing robot ...")
        self.respawn_robot(self.index)
    
        # self.get_logger().debug("Respawing goal ...")
        # self.respawn_goal(self.index)

        self.get_logger().debug("Environment reset performed ...")

    def respawn_goal(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            self.get_random_goal()
        else:
            self.get_goal(index)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")

        position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])+",z: "+str(0.01)+"}"
        pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.25)

    def get_goal(self, index):
        """
        """
        self.goal_pose = self.goals[index]
 
    def get_random_goal(self):
        """
        """
        if self.episode < 6 or self.episode % 25 == 0:
            x = 0.55
            y = 0.55
        else:
            x = random.randrange(-29, 29) / 10.0
            y = random.randrange(-29, 29) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]

        self.goal_pose = [x, y]

    def respawn_robot(self, index):
        """
        """
        if self.episode <= self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")
        logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")

        position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(0.07)+"}"
        orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
        pose = position+", "+orientation
        state = "'{state: {name: '"+self.robot_name+"',pose: {"+pose+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.25)

    def pause(self):
        """
        """
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service not available, waiting again...')
        future = self.pause_physics_client.call_async(req) 
        rclpy.spin_until_future_complete(self, future)

    def unpause(self):
        """
        """
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service not available, waiting again...')
        future = self.unpause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)


    def get_goal_pose(self):
        """
        """
        future = self.get_entity_client.send_request()

        x = future.state.pose.position.x
        y = future.state.pose.position.y

        # qz = future.state.pose.orientation.z
        # qw = future.state.pose.orientation.w
        # t1 = 2*(qw*qz)
        # t2 = 1 - 2*(qz*qz)
        # yaw = math.atan2(t1,t2)

        ############################################
        goal = PoseStamped()
        goal.pose.position.x = x
        goal.pose.position.y = y
        self.goal_pub.publish(goal)
        ############################################

        return [x, y]
