#!/usr/bin/env python3

import os
#import tensorflow as tf

import random
import sys
import time

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image

from ament_index_python.packages import get_package_share_directory

from rclpy.qos import qos_profile_sensor_data

from main_app.generic_sensor import Sensors

import numpy as np
import math
import subprocess
import json

from numpy import savetxt
import cv2
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile

class Pic4rlEnvironmentLidar(Node):
    def __init__(self):
        super().__init__('pic4rl_env_lidar')
        # To see debug logs
        #rclpy.logging.set_logger_level('pic4rl_env_lidar', 10)

        self.declare_parameters(namespace='',
        parameters=[
            ('data_path', '/root/gym_ws/src/PIC4rl_gym/training/pic4rl/goals_and_poses/indoor.json'),
            ('change_goal_and_pose', 2),
            ('timeout_steps', 600),
            ('collision_check', 0.40),
            ])

        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.change_episode = self.get_parameter('change_goal_and_pose').get_parameter_value().integer_value
        self.timeout_steps = self.get_parameter('timeout_steps').get_parameter_value().integer_value
        self.collision_check = self.get_parameter('collision_check').get_parameter_value().double_value

        self.goals, self.poses = self.get_goals_and_poses()

        self.get_logger().info("terminate getting goals")

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)
        self.sensors = Sensors(self)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos)

        # Initialise client
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        self.get_state_client = self.create_client(State, 'get_state')
        self.new_episode_client = self.create_client(Reset, 'new_episode')

        #self.timer = self.create_timer(0.1, self.get_state_callback)

        """##########
        State variables
        ##########"""
        self.change_index = -1
        self.init_step = True
        self.episode_step = 0
        self.goal_pos_x = None
        self.goal_pos_y = None
        self.previous_twist = None
        self.previous_pose = Odometry()

        #self.stage = 1
        self.lidar_points = 359
        self.cutoff = 5.0
        self.depth_image_raw = np.zeros((480,640), np.uint8)
        self.bridge = CvBridge()        
        #test variable
        self.step_flag = False
        self.twist_received = None
        
        self.robot_flag = False
        self.episode = 0
        #self.goals = [[-1.32, -4.0], [5.9, -1.40], [-5.57, 1.50]]
        #self.poses = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.2], [0.5, 0.8, -0.3], [0.8, -0.4, 0.0], [-0.6, 0.0, 0.5]]
        #self.state = {"scan": np.squeeze(np.ones((1,36))*15.0), "odom": [0.0,0.0,0.0]}

        """##########
        Environment initialization
        ##########"""

    """#############
    Main functions
    #############"""

    def get_goals_and_poses(self):
        data = json.load(open(self.data_path,'r'))
        return data["goals"], data["poses"]

    def render(self):

        pass

    def step(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])
        #twist.linear.y = float(action[1])
        twist.angular.z = float(action[1])
        observation, reward, done = self._step(twist)
        info = None
        return observation, reward, done, info

    def _step(self, twist=Twist(), reset_step = False):
        #After environment reset sensors data are not instaneously available
        #that's why there's the while. A timer could be added to increase robustness
        
        self.send_action(twist)
        # Get sensor_data
        sensor_data = self.get_sensor_data()

        lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_sensor_data(sensor_data)

        # Check events (failure,timeout, success)
        done, event = self.check_events(lidar_measurements, goal_distance, self.episode_step)

        if not reset_step:
            # Get reward
            reward = self.get_reward(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)
            observation = self.get_observation(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw)
        else:
            reward = None
            observation = None

        # Send observation and reward
        self.update_state(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)

        return  observation, reward, done
    
    def get_sensor_data(self):
        sensor_data = {}
        sensor_data["scan"] = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()
        if sensor_data["scan"] is None:
            sensor_data["scan"] = np.squeeze(np.ones((1,36))*15.0).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0,0.0,0.0]

        return sensor_data
    
    def reset(self, n_episode):
        #self.destroy_subscription('cmd_vel')
        self.episode = n_episode
        
        self.get_logger().debug("Environment reset ...")

        self.new_episode()
    
        self.get_logger().debug("Performing null step to reset variables")
        _,_,_, = self._step(reset_step = True)
        observation,_,_, = self._step()
        return observation

    def new_episode(self):
        #self.get_logger().debug("Resetting simulation ...")
        #self.reset_simulation()
        
        self.get_logger().debug("Resetting simulation ...")
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_simulation_client.call_async(req)
        
        if int(self.episode/self.change_episode) != self.change_index:
            self.change_index = int(self.episode/self.change_episode)

            self.get_logger().debug("Respawing robot ...")
            self.respawn_robot()
        
        self.get_logger().debug("Respawing goal ...")
        self.respawn_entity()

        self.get_logger().debug("Environment reset performed ...")
        #self.episode += 1

    """#############
    Secondary functions (used in main functions)
    #############"""
    def respawn_entity(self,):

        self.goal_x, self.goal_y = self.get_goal()
        #Goal initialization
        # Entity 'goal'
        self.entity_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.entity_dir_path = '/root/gym_ws/src/PIC4rl_gym/simulation/gazebo_sim/models/goal_box'
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'

        self.get_logger().debug("deleting goal entity...")
        try:
            self.delete_entity('goal')
        except:
            pass
        self.get_logger().debug("respawning goal entity...")
        entity_path=self.entity_path
        initial_pose = Pose()
        initial_pose.position.x = self.goal_pose_x
        initial_pose.position.y = self.goal_pose_y

        self.spawn_entity(initial_pose,self.entity_name,entity_path)

    def respawn_robot(self):
        if self.robot_flag:
            self.delete_entity("robot")
        self.robot_flag = False
        x, y , yaw = self.get_robot_pose()
        pose = '-x '+str(x)+' -y '+str(y)+' -Y '+str(yaw)
        robot = "robot "
        command = 'ros2 run gazebo_ros spawn_entity.py -entity '+robot+pose+' -topic /robot_description'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        self.robot_flag = True
        
    def get_robot_pose(self):

        index = int(np.random.uniform()*len(self.poses)) -1 
        self.x_robot = self.poses[index][0]
        self.y_robot = self.poses[index][1]
        self.yaw_robot = self.poses[index][2]

        self.get_logger().info("New robot pose: (x,y) : " + str(self.x_robot) + "," +str(self.y_robot))
        
        return self.x_robot, self.y_robot, self.yaw_robot
    
    def send_action(self,twist):
        #self.get_logger().debug("unpausing...")
        self.unpause()
        #self.get_logger().debug("publishing twist...")
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)
        #self.get_logger().debug("pausing...")
        self.pause()

    def process_sensor_data(self,sensor_data):

        #from Odometry msg to x,y, yaw, distance, angle wrt goal
        goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_odom(sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]

        return lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw

    def check_events(self, lidar_measurements, goal_distance, step):

        min_range = self.collision_check
        min_lidar = min(lidar_measurements)
        self.get_logger().debug("Min lidar: {}".format(min_lidar))

        if  0.05 <  min_lidar < min_range:
            # Collision
            self.get_logger().info('Collision')
            return True, "collision"

        if goal_distance < 0.30:
            # Goal reached
            self.get_logger().info('Goal')
            return True, "goal"

        if step >= self.timeout_steps:
            #Timeout
            self.get_logger().info('Timeout')
            print('step : ', step)
            return True, "timeout"

        return False, "None"

    def get_observation(self, twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw):

        #WITH LIDAR
        state_list = []
        state_list.append(float(goal_distance))
        state_list.append(float(goal_angle))

        for point in lidar_measurements:
            #state_list.append(float(point/self.max_obstacle_distance))
            #self.get_logger().info(point[0])
            state_list.append(float(point))
            #print(point)
        state = np.array(state_list,dtype = np.float32)
        #lidar_measurements = np.array(lidar_measurements, dtype = np.float32)
        #state = np.array([goal_distance, goal_angle, lidar_measurements], dtype = np.float32)
        return state

        

    def get_reward(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
        if self.episode < 200:
            yaw_reward = (1 - 2*math.sqrt(math.fabs(goal_angle / math.pi)))*0.6
        else:
            yaw_reward = (1 - 2*math.sqrt(math.fabs(goal_angle / math.pi)))*0.3
        distance_reward = 2*(2 * self.previous_goal_distance)
        #    (self.previous_goal_distance + goal_distance) - 1)
        #distance_reward = (2 - 2**(self.goal_distance / self.init_goal_distance))
        #yaw_reward = (1 - 2*math.sqrt(math.fabs(goal_angle / math.pi)))*0.8
        distance_reward = (self.previous_goal_distance - goal_distance)*35
        #v = twist.linear.x
        #w = twist.angular.z
        #speed_re = (3*v - math.fabs(w))

        reward =  yaw_reward + distance_reward 

        if event == "goal":
            reward += 1000
        elif event == "collision":
            reward += -100
        self.get_logger().debug(str(reward))

        return reward

    def get_def_goal(self):
        index = int(np.random.uniform()*len(self.goals)) -1
        self.goal_pose_x = self.goals[index][0]
        self.goal_pose_y = self.goals[index][1]
        self.get_logger().info("New goal: (x,y) : " + str(self.goal_pose_x) + "," +str(self.goal_pose_y))
        return self.goal_pose_x, self.goal_pose_y
 
    def get_goal(self):
        if self.episode < 6 or self.episode % 25==0:
            x = 0.4
            y = 0.4
        else:
            x = random.randrange(-29, 29) / 10.0
            y = random.randrange(-29, 29) / 10.0
        self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
        self.goal_pose_x = x
        self.goal_pose_y = y
        return x,y

    def update_state(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
        #Here state variables are updated
        self.episode_step += 1
        self.previous_twist = twist
        self.previous_lidar_measurements = lidar_measurements
        self.previous_goal_distance = goal_distance
        self.previous_goal_angle = goal_angle
        self.previous_pos_x = pos_x
        self.previous_pos_y = pos_y
        self.previous_yaw = yaw
        # If done, set flag for resetting everything at next step
        if done:
            self.init_step = True
            self.episode_step = 0

    """#############
    Auxiliar functions (used in secondary functions)
    #############"""

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

    def process_odom(self, odom):
        #self.previous_pose.pose.pose.position.x = odom_msg.pose.pose.position.x
        #self.previous_pose.pose.pose.position.y = odom_msg.pose.pose.position.y

        pos_x = odom[0]
        pos_y = odom[1]
        yaw = odom[2]

        goal_distance = math.sqrt(
            (self.goal_pose_x-pos_x)**2
            + (self.goal_pose_y-pos_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-pos_y,
            self.goal_pose_x-pos_x)

        goal_angle = path_theta - yaw

        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

        return goal_distance, goal_angle, pos_x, pos_y, yaw
    
    def spawn_entity(self,pose = None, name = None, entity_path = None, entity = None):
        if not pose:
            pose = Pose()
        req = SpawnEntity.Request()
        req.name = name
        if entity_path:
            entity = open(entity_path, 'r').read()
        req.xml = entity
        req.initial_pose = pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)
        
    def respawn_goal(self):

        #Goal initialization
        self.entity_path = os.path.join(get_package_share_directory("pic4rl"), 'models', 'goal_box', 'model.sdf')

        self.entity_name = 'goal'

        self.get_logger().debug("deleting goal entity...")
        try:
            self.delete_entity('goal')
        except:
            pass
        self.get_logger().debug("respawning goal entity...")

        initial_pose = Pose()
        initial_pose.position.x = self.goal_pose_x
        initial_pose.position.y = self.goal_pose_y

        self.spawn_entity(initial_pose,self.entity_name,self.entity_path)

    def delete_entity(self, entity_name):
        req = DeleteEntity.Request()
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.delete_entity_client.call_async(req)
        self.get_logger().debug('Entity deleting request sent ...')
