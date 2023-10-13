import math
import time
import random
import logging
import subprocess
import numpy as np

from pic4rl.utils.train_utils import EnvUtils
from pic4rl.sensors import Sensors

import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist


class Pic4rlEnv_Vineyard(py_environment.PyEnvironment, Node):
    """
    """
    def __init__(self, env_name="Generic_Pic4rlEnv_Vineyard"):
        """
        """
        py_environment.PyEnvironment.__init__(self)
        Node.__init__(self, f'pic4rlenv_{env_name}')

        self.env_name   = env_name
        self.sensors    = Sensors(self)
        self.utils      = EnvUtils()

        params  = self.utils.get_ros_params(
            'pic4rl', 'config/params.yaml', 'params')

        goals   = self.utils.get_data(
            'pic4rl', 'data/' + params.data_name + '.yaml', 'goals')
        poses   = self.utils.get_data(
            'pic4rl', 'data/' + params.data_name + '.yaml', 'poses')

        self.spin_sensors_callbacks()

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        self.reset_world_client     = self.create_client(
            Empty, 'reset_world')

        self.define_env(params)
        self.goals = np.array(goals, dtype=np.float32)
        self.poses = np.array(poses, dtype=np.float32)
        self.params = params

        self.reset_var(total=True)

        logging.debug(f"{self.env_name}: PIC4RL_Environment: Starting process")

    def define_env(self, params):
        """
        """
        min_action = np.array([params.min_lin_vel, params.min_ang_vel])
        max_action = np.array([params.max_lin_vel, params.max_ang_vel])

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), 
            dtype=np.float32, 
            minimum=np.array(min_action, dtype=np.float32),
            maximum=np.array(max_action, dtype=np.float32), 
            name='action'
        )

        state =[
        [params.min_lin_vel, params.max_lin_vel], # goal_distance 
        [params.min_ang_vel, params.max_ang_vel], # goal angle or yaw
        ]
        if params.visual_data == 'features':
            for i in range(params.features):
                state = state + [[0., params.dist_cutoff]]
        elif params.visual_data == 'image':
            raise Exception("Sorry, not supported yet") 
        state = np.array(state)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(state),), 
            dtype=np.float32, 
            minimum=np.array(state[:,0], dtype=np.float32),
            maximum=np.array(state[:,1], dtype=np.float32),
            name='observation'
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def spin_sensors_callbacks(self):
        """
        """
        rclpy.spin_once(self)
        while None in self.sensors.sensor_msg.values():
            rclpy.spin_once(self)
        self.sensors.sensor_msg = dict.fromkeys(self.sensors.sensor_msg.keys(), None)
        
    def reset_var(self, total=False):
        """
        """
        if total:
            self.episode  = 0
            
        else:
            self.episode  += 1
            
        self.n_step     = 0
        self.cum_rew    = 0
        self.done       = False
        self.collision_count = 0

        self.new_episode()

        self._step(reset_step=True)
        self.previous_state = np.copy(self.state)

    def _reset(self):
        self.reset_var()
        
        return ts.restart(np.array(self.state, dtype=np.float32))
    
    def _step(self, action=np.array([]), reset_step=False):
        if self.done:
            return self._reset()
        
        self.n_step += 1 
        
        self.send_action(action)

        self.spin_sensors_callbacks()
        lidar_measurements, depth_image, goal_info, robot_pose, collision = self.get_sensor_data()

        self.done, event = self.check_events(goal_info, collision)

        self.state = self.get_observation(action, depth_image, goal_info)

        if reset_step:
            reward = 0
        else:
            reward = self.get_reward(action, goal_info, event)
        
        self.update(action, depth_image, goal_info, robot_pose)
        
        if self.done:
            return ts.termination(
                np.array(self.state, dtype=np.float32), 
                reward=reward
            )
        
        else:
            return ts.transition(
                np.array(self.state, dtype=np.float32), 
                reward=reward, 
                discount=1.0
            )
        
    def send_action(self, action, action_step=1):
        """
        """
        twist = Twist()

        if action.size > 0:
            twist.linear.x = float(action[0])
            twist.angular.z = float(action[1])

        self.cmd_vel_pub.publish(twist)

    def get_sensor_data(self):
        """
        """
        sensor_data = {}
        sensor_data["scan"], collision, _ = self.sensors.get_laser()
        sensor_data["odom"] = self.sensors.get_odom()
        sensor_data["depth"] = self.sensors.get_depth()
        
        if sensor_data["scan"] is None:
            sensor_data["scan"] = (
                np.ones(self.params.lidar_points
                    )*self.params.lidar_distance).tolist()
        if sensor_data["odom"] is None:
            sensor_data["odom"] = [0.0,0.0,0.0]
        if sensor_data["depth"] is None:
            sensor_data["depth"] = np.ones(
                (self.params.depth_height,self.params.depth_width,1)
                )*self.params.dist_cutoff

        goal_info, robot_pose = self.utils.process_odom(
            self.goal_pose, sensor_data["odom"])
        lidar_measurements = sensor_data["scan"]
        depth_image = sensor_data["depth"]

        return lidar_measurements, depth_image, goal_info, robot_pose, collision

    def check_events(self, goal_info, collision):
        """
        """
        if collision:
            self.collision_count += 1
            if self.collision_count >= 3:
                self.collision_count = 0
                logging.info(f"{self.env_name}: Ep {'evaluate' if self.params.evaluate else self.episode+1}: Collision")
                return True, "collision"
            else:
                return False, "collision"

        if goal_info[0] < self.params.goal_tolerance:
            logging.info(f"{self.env_name}: Ep {'evaluate' if self.params.evaluate else self.episode+1}: Goal")
            return True, "goal"

        if self.n_step+1 == self.params.ep_max_steps:
            logging.info(f"{self.env_name}: Ep {'evaluate' if self.params.evaluate else self.episode+1}: Timeout")
            return True, "timeout"

        return False, "None"

    def get_reward(self, action, goal_info, event):
        """
        """
        yaw_reward = (1 - 2*math.sqrt(math.fabs(goal_info[1] / math.pi)))*0.6
        #y_reward = (-2**(math.fabs(0.45 - 2*robot_pose[1]))+1)*10
        #distance_reward = 2*((2 * self.previous_goal_distance) / \
        #   (self.previous_goal_distance + goal_distance) - 1)
        #distance_reward = (2 - 2**(self.goal_distance / self.init_goal_distance))
        distance_reward = (self.previous_goal_info[0] - goal_info[0])*35

        speed_reward = (action[0] - math.fabs(w))
        
        reward =  yaw_reward + distance_reward + speed_reward

        if event == "goal":
            reward += 1000
        if event == "collision":
            #reward += -1000*math.fabs(v)**2
            reward = -500
        if event == "reverse":
            reward = -500
        else:
            reward += -1

        logging.debug(str(reward))

        return reward

    def get_observation(self, action, depth_image, goal_info):
        """
        """
        if action.size == 0:
            action = np.array([0., 0.], dtype=np.float32)
        features    = depth_image.flatten()
        state = np.concatenate((action, features))

        return state

    def update(self, action, depth_image, goal_info, robot_pose):
        """
        """
        self.previous_action = action
        self.previous_depth_image = depth_image
        self.previous_goal_info = goal_info
        self.previous_robot_pose = robot_pose

    def new_episode(self):
        """
        """
        req = Empty.Request()

        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            logging.warn(f'{self.env_name}: service not available, waiting again...')
        self.reset_world_client.call_async(req)
        
        if self.episode % self.params.change_ep == 0. or self.params.evaluate:
            self.index = int(np.random.uniform()*len(self.poses)) -1 

        self.respawn_robot(self.index)
    
        self.respawn_goal(self.index)

        logging.debug(f"{self.env_name}: Environment reset performed ...")

    def respawn_goal(self, index):
        """
        """
        if self.episode <= self.params.starting_eps:
            self.goal_pose = self.utils.get_random_goal(self.episode, self.poses[0])
        else:
            self.goal_pose = self.goals[index]

        logging.info(f"{self.env_name}: Ep {'evaluate' if self.params.evaluate else self.episode+1} goal pose [x, y]: {self.goal_pose}")
        
        position = "{x: "+str(self.goal_pose[0])+",y: "+str(self.goal_pose[1])+",z: "+str(0.01)+"}"
        pose = "'{state: {name: 'goal',pose: {position: "+position+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+pose,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.25)

    def respawn_robot(self, index):
        """
        """
        if self.episode <= self.params.starting_eps:
            x, y, yaw = tuple(self.poses[0])
        else:
            x, y , yaw = tuple(self.poses[index])

        qz = np.sin(yaw/2)
        qw = np.cos(yaw/2)

        logging.info(f"{self.env_name}: Ep {'evaluate' if self.params.evaluate else self.episode+1} robot pose [x,y,yaw]: {[x, y, yaw]}")

        position = "position: {x: "+str(x)+",y: "+str(y)+",z: "+str(0.07)+"}"
        orientation = "orientation: {z: "+str(qz)+",w: "+str(qw)+"}"
        pose = position+", "+orientation
        state = "'{state: {name: '"+self.params.robot_name+"',pose: {"+pose+"}}}'"
        subprocess.run(
            "ros2 service call /test/set_entity_state gazebo_msgs/srv/SetEntityState "+state,
            shell=True,
            stdout=subprocess.DEVNULL
            )
        time.sleep(0.25)

# rclpy.init()
# environment = Pic4rlEnv()
# utils.validate_py_environment(environment, episodes=5)