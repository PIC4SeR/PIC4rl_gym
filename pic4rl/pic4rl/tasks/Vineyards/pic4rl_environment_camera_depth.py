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
from pic4rl.utils.env_utils import *
from pic4rl_testing.nav_metrics import Navigation_Metrics


class Pic4rlEnvironmentCamera(Node):
    def __init__(
        self,
    ):
        """ """
        super().__init__("pic4rl_training_vineyard_camera")
        self.declare_parameter("package_name", "pic4rl")
        self.package_name = (
            self.get_parameter("package_name").get_parameter_value().string_value
        )
        goals_path = os.path.join(
            get_package_share_directory(self.package_name), "goals_and_poses"
        )
        main_params_path = os.path.join(
            get_package_share_directory(self.package_name), "config", "main_params.yaml"
        )
        train_params_path = os.path.join(
            get_package_share_directory(self.package_name),
            "config",
            "training_params.yaml",
        )
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), "models/goal_box/model.sdf"
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
                ("mode", main_params["mode"]),
                ("data_path", main_params["data_path"]),
                ("change_goal_and_pose", train_params["--change_goal_and_pose"]),
                ("starting_episodes", train_params["--starting_episodes"]),
                ("timeout_steps", train_params["--episode-max-steps"]),
                ("robot_name", main_params["robot_name"]),
                ("goal_tolerance", main_params["goal_tolerance"]),
                ("visual_data", main_params["visual_data"]),
                ("features", main_params["features"]),
                ("channels", main_params["channels"]),
                ("image_width", main_params["depth_param"]["width"]),
                ("image_height", main_params["depth_param"]["height"]),
                ("dist_cutoff", main_params["depth_param"]["dist_cutoff"]),
                ("lidar_dist", main_params["laser_param"]["max_distance"]),
                ("lidar_points", main_params["laser_param"]["num_points"]),
                ("update_frequency", main_params["update_frequency"]),
            ],
        )

        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.data_path = (
            self.get_parameter("data_path").get_parameter_value().string_value
        )
        self.data_path = os.path.join(goals_path, self.data_path)
        self.change_episode = (
            self.get_parameter("change_goal_and_pose")
            .get_parameter_value()
            .integer_value
        )
        self.starting_episodes = (
            self.get_parameter("starting_episodes").get_parameter_value().integer_value
        )
        self.timeout_steps = (
            self.get_parameter("timeout_steps").get_parameter_value().integer_value
        )
        self.robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        self.goal_tolerance = (
            self.get_parameter("goal_tolerance").get_parameter_value().double_value
        )
        self.visual_data = (
            self.get_parameter("visual_data").get_parameter_value().string_value
        )
        self.features = (
            self.get_parameter("features").get_parameter_value().integer_value
        )
        self.channels = (
            self.get_parameter("channels").get_parameter_value().integer_value
        )
        self.image_width = (
            self.get_parameter("image_width").get_parameter_value().integer_value
        )
        self.image_height = (
            self.get_parameter("image_height").get_parameter_value().integer_value
        )
        self.max_depth = (
            self.get_parameter("dist_cutoff").get_parameter_value().double_value
        )
        self.lidar_distance = (
            self.get_parameter("lidar_dist").get_parameter_value().double_value
        )
        self.lidar_points = (
            self.get_parameter("lidar_points").get_parameter_value().integer_value
        )
        self.params_update_freq = (
            self.get_parameter("update_frequency").get_parameter_value().double_value
        )

        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)

        self.logdir = create_logdir(
            train_params["--policy"], main_params["sensor"], train_params["--logdir"]
        )
        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", qos)

        self.reset_world_client = self.create_client(Empty, "reset_world")
        self.pause_physics_client = self.create_client(Empty, "pause_physics")
        self.unpause_physics_client = self.create_client(Empty, "unpause_physics")

        self.episode_step = 0
        self.previous_twist = Twist()
        self.episode = 0
        self.collision_count = 0
        self.t0 = 0.0
        self.evaluate = False

        # if want to use a DWA to provide demonstrative exploration episodes
        self.explore_demo = 15
        # self.controller = Pic4DWA()

        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]

        self.get_logger().info(f"Gym mode: {self.mode}")
        if self.mode == "testing":
            self.nav_metrics = Navigation_Metrics(main_params, self.logdir)

        self.get_logger().debug("PIC4RL_Environment: Starting process")

    def step(self, action, episode_step=0):
        """ """
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.episode_step = episode_step

        observation, reward, done = self._step(twist)
        info = None

        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step=False):
        """ """
        self.get_logger().debug("sending action...")
        self.send_action(twist)

        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        (
            lidar_measurements,
            depth_image,
            goal_info,
            robot_pose,
            collision,
            _,
        ) = self.get_sensor_data()

        if self.mode == "testing":
            self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step)

        self.get_logger().debug("checking events...")
        done, event = self.check_events(
            lidar_measurements, goal_info, robot_pose, collision
        )

        if not reset_step:
            self.get_logger().debug("getting reward...")
            reward = self.get_reward(
                twist, lidar_measurements, goal_info, robot_pose, done, event
            )

            self.get_logger().debug("getting observation...")
            observation = self.get_observation(
                twist, depth_image, goal_info, robot_pose
            )
        else:
            reward = None
            observation = None

        # Send observation and reward
        self.update_state(twist, depth_image, goal_info, robot_pose, done, event)
        if done:
            time.sleep(1.5)

        return observation, reward, done

    def get_goals_and_poses(self):
        """ """
        data = json.load(open(self.data_path, "r"))

        return data["initial_pose"], data["goals"], data["poses"]

    def spin_sensors_callbacks(self):
        """ """
        self.get_logger().debug("spinning node...")
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
            self.get_logger().debug("spin once ...")
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def send_action(self, twist):
        """ """
        # self.get_logger().debug("unpausing...")
        # self.unpause()

        # self.get_logger().debug("publishing twist...")
        self.cmd_vel_pub.publish(twist)

        # Regulate frequency of send action if needed
        freq, t1 = compute_frequency(self.t0)
        t0 = t1
        frequency_control(self.params_update_freq)

        # self.get_logger().debug("pausing...")
        # self.pause()

    def get_sensor_data(self):
        """ """
        sensor_data = {"depth": None}
        sensor_data["scan"], collision, complete_laser_data = self.sensors.get_laser()
        sensor_data["odom"], velocities = self.sensors.get_odom(vel=True)
        sensor_data["depth"] = self.sensors.get_depth()

        if sensor_data["scan"] is None:
            sensor_data["scan"] = (
                np.ones(self.lidar_points) * self.lidar_distance
            ).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0, 0.0, 0.0]
        if sensor_data["depth"] is None:
            sensor_data["depth"] = (
                np.ones((self.image_height, self.image_width, 1)) * self.cutoff
            )

        self.get_logger().debug("processing odom...")
        goal_info, robot_pose = process_odom(self.goal_pose, sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]
        depth_image = sensor_data["depth"]

        return (
            lidar_measurements,
            depth_image,
            goal_info,
            robot_pose,
            collision,
            velocities,
        )

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
        """ """
        # FOR VINEYARD ONLY ##
        # if math.fabs(robot_pose[2]) > 1.57:
        #     robot_pose[2] = math.fabs(robot_pose[2]) - 3.14
        # #print('robot pose :', robot_pose)

        # yaw_limit = math.fabs(robot_pose[2])-1.4835  #check yaw is less than 85°
        # self.get_logger().debug("Yaw limit: {}".format(yaw_limit))

        # if yaw_limit > 0:
        #     self.get_logger().info('Reverse: yaw too high')
        #     return True, "reverse"

        # if collision:
        #     self.collision_count += 1
        #     if self.collision_count >= 3:
        #         self.collision_count = 0
        #         self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
        #         logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Collision")
        #         return True, "collision"
        #     else:
        #         return False, "None"

        if goal_info[0] < self.goal_tolerance:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal"
            )
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            return True, "goal"

        if self.episode_step + 1 >= self.timeout_steps:
            self.get_logger().info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            logging.info(
                f"Ep {'evaluate' if self.evaluate else self.episode+1}: Timeout"
            )
            return True, "timeout"

        return False, "None"

    def get_reward(self, twist, lidar_measurements, goal_info, robot_pose, done, event):
        """ """
        yaw_reward = (1 - 3 * math.sqrt(math.fabs(goal_info[1] / math.pi))) * 0.1
        # y_reward = (-2**(math.fabs(0.45 - 2*robot_pose[1]))+1)*10
        # distance_reward = 2*((2 * self.previous_goal_distance) / \
        #   (self.previous_goal_distance + goal_distance) - 1)
        # distance_reward = (2 - 2**(self.goal_distance / self.init_goal_distance))
        distance_reward = (self.previous_goal_info[0] - goal_info[0]) * 5
        v = twist.linear.x
        w = twist.angular.z
        # speed_reward = (v - math.fabs(w))

        reward = yaw_reward + distance_reward

        if event == "goal":
            reward = 300
        elif event == "collision":
            # reward = -1000*math.fabs(v)**2
            reward = -100
        elif event == "reverse":
            reward = -200
        else:
            reward += -0.1

        # print("yaw reward: ", yaw_reward)
        # print("speed reward: ", speed_reward)
        # print("distance_reward: ", distance_reward)
        # print("reward_tot ", reward)
        return reward

    def get_observation(self, twist, depth_image, goal_info, robot_pose):
        """ """
        # flattened depth image
        if self.visual_data == "features":
            features = depth_image.flatten()

        # previous velocity state
        v = twist.linear.x
        w = twist.angular.z
        vel = np.array([v, w], dtype=np.float32)
        state = np.concatenate((vel, features))
        return state

    def update_state(self, twist, depth_image, goal_info, robot_pose, done, event):
        """ """
        self.previous_twist = twist
        self.previous_depth_image = depth_image
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

    def reset(self, n_episode, tot_steps, evaluate=False):
        """ """
        if self.mode == "testing":
            self.nav_metrics.calc_metrics(n_episode, self.initial_pose, self.goal_pose)
            self.nav_metrics.log_metrics_results(n_episode)
            self.nav_metrics.save_metrics_results(n_episode)

        self.episode = n_episode
        self.evaluate = evaluate
        logging.info(
            f"Total_episodes: {n_episode}{' evaluation episode' if self.evaluate else ''}, Total_steps: {tot_steps}, episode_steps: {self.episode_step+1}\n"
        )
        print()
        self.get_logger().info("Initializing new episode ...")
        logging.info("Initializing new episode ...")

        self.get_logger().debug("pausing...")
        self.pause()

        self.new_episode()

        self.get_logger().debug("unpausing...")
        self.unpause()
        self.get_logger().debug("Performing null step to reset variables")
        self.episode_step = 0

        (
            _,
            _,
            _,
        ) = self._step(reset_step=True)
        (
            observation,
            _,
            _,
        ) = self._step()

        # if self.episode < 50 or (self.episode % self.explore_demo == 0.) and not self.evaluate:
        #    exploration_ep = True
        #    self.get_logger().info("Pseudo-Demonstrative exploration episode ...")
        # else:
        exploration_ep = False

        return observation, exploration_ep

    def new_episode(self):
        """ """

        # self.get_logger().debug("Resetting simulation ...")
        # req = Empty.Request()

        # while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().warn('service not available, waiting again...')
        # self.reset_world_client.call_async(req)

        if self.episode % self.change_episode == 0.0 or self.evaluate:
            self.index = int(np.random.uniform() * len(self.poses)) - 1

        self.get_logger().debug("Respawing robot ...")
        self.respawn_robot(self.index)

        self.get_logger().debug("Respawing goal ...")
        self.respawn_goal(self.index)

        time.sleep(1.0)
        self.get_logger().debug("Environment reset performed ...")

    def respawn_goal(self, index):
        """ """
        self.get_goal(index)

        self.get_logger().info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}"
        )
        logging.info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}"
        )

        # position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])+",z: "+str(0.01)+"}"
        # pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}'"
        # subprocess.run(
        #     "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
        #     shell=True,
        #     stdout=subprocess.DEVNULL
        #     )
        # time.sleep(0.25)

    def get_goal(self, index):
        """ """
        self.goal_pose = self.goals[index]

    def respawn_robot(self, index):
        """ """
        if self.episode < self.starting_episodes:
            x, y, yaw = tuple(self.initial_pose)
        else:
            x, y, yaw = tuple(self.poses[index])

        qz = np.sin(yaw / 2)
        qw = np.cos(yaw / 2)

        self.get_logger().info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}"
        )
        logging.info(
            f"Ep {'evaluate' if self.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}"
        )

        position = "position: {x: " + str(x) + ",y: " + str(y) + ",z: " + str(0.1) + "}"
        orientation = "orientation: {z: " + str(qz) + ",w: " + str(qw) + "}"
        pose = position + ", " + orientation
        state = "'{state: {name: '" + self.robot_name + "',pose: {" + pose + "}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "
            + state,
            shell=True,
            stdout=subprocess.DEVNULL,
        )
        time.sleep(1.0)

    def pause(self):
        """ """
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        future = self.pause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def unpause(self):
        """ """
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")
        future = self.unpause_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    # def call_DWA(self):
    #     """
    #     """
    #     self.spin_sensors_callbacks()
    #     _, _, _, robot_pose, collision, velocities = self.get_sensor_data()

    #     obs = self.dwa_process_lidar(self.laser_data, robot_pose)

    #     x, ob, goal = self.controller.get_env_data(robot_pose, velocities, obs, self.goal_pose)
    #     dw = self.controller.calc_dynamic_window(x)
    #     u, trajectory = self.controller.calc_control_and_trajectory(x, dw, goal, ob)

    #     return u

    # def dwa_process_lidar(self, lidar_measurements, robot_pose):
    #     """
    #     """
    #     actual_angle = self.laser_info[0]
    #     increment = self.laser_info[2]
    #     ob = []
    #     ob_points = []

    #     for point in lidar_measurements:
    #         p = [point*math.cos(actual_angle), point*math.sin(actual_angle)]
    #         p = tf_decompose(robot_pose,[p[0], p[1], 0.0, 1.0])

    #         ob.append([p[0], p[1]])
    #         actual_angle += increment

    #     return ob
