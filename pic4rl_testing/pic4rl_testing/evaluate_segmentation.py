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
import yaml
import logging
import datetime
from pathlib import Path

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose,PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from pic4rl.sensors import Sensors

from rclpy.parameter import Parameter
#from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rcl_interfaces.msg import ParameterDescriptor, ParameterValue

from pic4rl_testing.nav_metrics import Navigation_Metrics

class EvaluateNav(Node):
    def __init__(self):
        super().__init__('evaluate_nav')
        
        #rclpy.logging.set_logger_level('evaluate_nav', 10)
        goals_path      = os.path.join(
            get_package_share_directory('pic4rl_testing'), 'goals_and_poses')
        main_params_path  = os.path.join(
            get_package_share_directory('pic4rl_testing'), 'config', 'main_params.yaml')
        train_params_path= os.path.join(
            get_package_share_directory('pic4rl_testing'), 'config', 'training_params.yaml')
        self.entity_path = os.path.join(
            get_package_share_directory("gazebo_sim"), 
            'models/goal_box/model.sdf'
            )
        
        with open(main_params_path, 'r') as file:
            main_params = yaml.safe_load(file)['main_node']['ros__parameters']
        with open(train_params_path, 'r') as train_param_file:
            train_params = yaml.safe_load(train_param_file)['training_params']

        self.declare_parameters(
            namespace   = '',
            parameters  = [
                ('data_path', main_params['data_path']),
                ('n_experiments', train_params['--n-experiments']),
                ('change_goal_and_pose', train_params['--change_goal_and_pose']),
                ('starting_episodes', train_params['--starting_episodes']),
                ('timeout_steps', train_params['--episode-max-steps']),
                ('robot_name', main_params['robot_name']),
                ('goal_tolerance', main_params['goal_tolerance']),
                ('update_frequency', main_params['update_frequency']),
                ('lidar_dist', main_params['laser_param']['max_distance']),
                ('lidar_points', main_params['laser_param']['num_points'])
                ]
            )

        self.data_path      = self.get_parameter(
            'data_path').get_parameter_value().string_value
        self.data_path      = os.path.join(goals_path, self.data_path)
        self.n_experiments = self.get_parameter(
            'n_experiments').get_parameter_value().integer_value
        self.change_episode = self.get_parameter(
            'change_goal_and_pose').get_parameter_value().integer_value
        self.starting_episodes = self.get_parameter(
            'starting_episodes').get_parameter_value().integer_value
        self.timeout_steps  = self.get_parameter(
            'timeout_steps').get_parameter_value().integer_value
        self.robot_name     = self.get_parameter(
            'robot_name').get_parameter_value().string_value
        self.goal_tolerance     = self.get_parameter(
            'goal_tolerance').get_parameter_value().double_value
        self.params_update_freq   = self.get_parameter(
            'update_frequency').get_parameter_value().double_value
        self.lidar_distance = self.get_parameter(
            'lidar_dist').get_parameter_value().double_value
        self.lidar_points   = self.get_parameter(
            'lidar_points').get_parameter_value().integer_value

        # create Sensor class to get and process sensor data
        self.sensors = Sensors(self)
        self.crop_name = 'vineyard_straight'
        self.true_path = 6.4
        self.create_logdir(train_params['--logdir'])
        self.spin_sensors_callbacks()

        qos = QoSProfile(depth=10)
        # init goal publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            'goal_pose',
            qos)


        # create reset world client 
        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        self.entity_path = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')

        self.episode_step = 0
        self.episode = 0
        self.collision_count = 0
        self.t0 = 0.0
        self.start_pose = [0., 0., 0.]
        self.index = 0
    
        self.initial_pose, self.goals, self.poses = self.get_goals_and_poses()
        self.goal_pose = self.goals[0]

        self.nav_metrics = Navigation_Metrics(main_params, self.logdir)

        self.get_logger().info("Evaluate Navigation: Starting process")
        self.get_logger().info("Navigation metrics collected at: " + str(self.params_update_freq)+' Hz')

    def step(self, episode_step=0):
        """
        """
        self.get_logger().debug("Env step : " + str(episode_step))
        self.episode_step = episode_step

        done = self._step()

        return done

    def _step(self, reset_step = False):
        """
        """
        self.get_logger().debug("getting sensor data...")
        self.spin_sensors_callbacks()
        lidar_measurements, goal_info, robot_pose, collision = self.get_sensor_data()

        self.get_logger().debug("checking events...")
        done, event = self.check_events(lidar_measurements, goal_info, robot_pose, collision)

        self.get_logger().debug("compute metrics...")
        self.nav_metrics.get_metrics_data(lidar_measurements, self.episode_step, done=done)

        self.compute_frequency()
        self.frequency_control()

        self.update_state(lidar_measurements, goal_info, robot_pose, done, event)

        return done

    def spin_sensors_callbacks(self):
        """
        """
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)

    def compute_frequency(self,):
        t1=time.perf_counter()
        step_time = t1-self.t0
        self.t0 = t1
        action_hz = 1./(step_time)
        self.get_logger().debug('Sending action at '+str(action_hz))

    def frequency_control(self):
        #self.get_logger().debug("Sleeping for: "+str(1/self.params_update_freq) +' s')
        time.sleep(1/self.params_update_freq)

    def get_sensor_data(self):
        """
        """
        sensor_data = {}
        sensor_data["scan"], collision = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()
    
        if sensor_data["scan"] is None:
            self.get_logger().debug("scan data is None...")
            sensor_data["scan"] = np.squeeze(np.ones((1,self.lidar_points))*2.0).tolist()
            collision = False
        if sensor_data["odom"] is None:
            self.get_logger().debug("odom data is None...")
            sensor_data["odom"] = [0.0,0.0,0.0]

        goal_info, robot_pose = self.process_odom(sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]

        return lidar_measurements, goal_info, robot_pose, collision

    def process_odom(self, odom):

        goal_distance = math.sqrt(
            (self.goal_pose[0]-odom[0])**2
            + (self.goal_pose[1]-odom[1])**2)

        path_theta = math.atan2(
            self.goal_pose[1]-odom[1],
            self.goal_pose[0]-odom[0])

        goal_angle = path_theta - odom[2]

        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        goal_info = [goal_distance, goal_angle]
        robot_pose = [odom[0], odom[1], odom[2]]

        return goal_info, robot_pose

    def check_events(self, lidar_measurements, goal_info, robot_pose, collision):
    
        if goal_info[0] < self.goal_tolerance:
            self.get_logger().info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            logging.info(f"Ep {'evaluate' if self.evaluate else self.episode+1}: Goal")
            return True, "goal"

        # check collision
        if  collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                self.get_logger().info('Collision')
                logging.info(f"Test Episode {self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        # check timeout steps
        if self.episode_step >= self.timeout_steps:
            self.get_logger().info('Timeout. Step: '+str(self.episode_step))
            logging.info(f"Test Episode {self.episode+1}: Timeout")
            return True, "timeout"

        return False, "None"


    def get_goals_and_poses(self):
        """
        """
        data = json.load(open(self.data_path,'r'))

        return data["initial_pose"], data["goals"], data["poses"]

    def update_state(self,lidar_measurements, goal_info, robot_pose, done, event):
        """
        """
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

    def reset(self, episode):
        """
        """
        self.episode = episode

        self.get_logger().info(f"Initializing new episode: scenario {self.index}")
        logging.info(f"Initializing new episode: scenario {self.index}")
        self.new_episode()
        self.get_logger().debug("Performing null step to reset variables")

        _ = self._step(reset_step = True)
    
    def new_episode(self):
        """
        """
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()

        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_world_client.call_async(req)
        
        self.get_logger().debug("Respawning robot ...")
        self.respawn_robot(self.index)
    
        self.get_logger().debug("Respawning goal ...")
        self.respawn_goal(self.index)

        self.index = self.index+1 if self.index<len(self.poses)-1 else 0 

        self.get_logger().debug("Environment reset performed ...")

    def respawn_robot(self, index):
        """
        """
        x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        self.start_pose = [x,y,yaw]

        self.get_logger().info(f"Test Episode {self.episode+1}, robot pose [x,y,yaw]: {self.start_pose}")
        logging.info(f"Test Episode {self.episode+1}, robot pose [x,y,yaw]: {self.start_pose}")

        position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(0.07)+"}"
        orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
        pose = position+", "+orientation
        state = "'{state: {name: '"+self.robot_name+"',pose: {"+pose+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
            shell=True,
            stdout=subprocess.DEVNULL
            )

    def respawn_goal(self, index):
        """
        """
        self.get_goal(index)

        self.get_logger().info(f"Test Episode {self.episode+1}, goal pose [x, y]: {self.goal_pose}")
        logging.info(f"Test Episode {self.episode+1}, goal pose [x, y]: {self.goal_pose}")

        position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])+",z: "+str(0.01)+"}"
        pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}'"

        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )

    def get_goal(self, index):
        # get goal from predefined list
        self.goal_pose = self.goals[index]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose))

    def send_goal(self,pose):
        # Set the robot's goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'odom'
        goal_pose.pose.position.x = pose[0]
        goal_pose.pose.position.y = pose[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)
        
    def get_random_goal(self):

        x = random.randrange(-29, 38) / 10.0
        y = random.randrange(-38, 38) / 10.0

        x += self.initial_pose[0]
        y += self.initial_pose[1]
            
        self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
        self.goal_pose = [x, y]

    def pause(self):

        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pause_physics_client.call_async(req) 

    def unpause(self):

        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.unpause_physics_client.call_async(req)


    def create_logdir(self, logdir):
            self.log_folder_name = f"Segmentation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{self.robot_name}_{self.crop_name}/"
            self.logdir = os.path.join(logdir, self.log_folder_name)
            Path(self.logdir).mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(self.logdir, 'screen_logger.log'), 
                level=logging.INFO)

    def evaluate(self,):
        for n in range(self.n_experiments):
            for episode in range(len(self.goals)):
                episode_steps = 0
                self.reset(episode)
                done = False
                while not done:
                    done = self.step(episode_steps)
                    episode_steps += 1
                self.nav_metrics.calc_metrics(self.episode, self.start_pose, self.goal_pose, self.true_path)
                self.nav_metrics.log_metrics_results(self.episode)
                self.nav_metrics.save_metrics_results(self.episode)
                return


def main(args=None):
    rclpy.init(args=args)

    eval_node = EvaluateNav()
    try:
        eval_node.evaluate()
    except Exception as e:
            eval_node.get_logger().error("Error in starting nav evaluation node: {}".format(e))
    
    eval_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()