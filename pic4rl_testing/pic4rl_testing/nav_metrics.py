#!/usr/bin/env python3

# Python libraries
import os
import time
import math
import yaml
import numpy as np
import logging

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState

def tf_compose(robot_pose, goal_pose):
    """
    This method composes two consecutive reference frames.
    For example, it can return the pose of the goal, converted from 
    fixed reference frame to robot reference frame, with robot_pose and 
    goal_pose given in fixed reference frame.
    """
    xr, yr, zr = tuple(robot_pose)
    goal_pose = np.concatenate((goal_pose, [0., 1.]), axis=0)

    A = np.zeros([4,4], dtype=float)
    A[0,0] = np.cos(zr)
    A[1,0] = np.sin(zr)
    A[0,1] = -A[1,0]
    A[1,1] = A[0,0]
    A[0,3] = xr
    A[1,3] = yr
    A[2,3] = 0
    A[3,3] = 1
    A[2,2] = 1
    A = np.linalg.inv(A)
    
    return np.matmul(A, goal_pose)

def quat_to_euler(qz, qw):
    """
    """
    t1 = 2*(qw*qz)
    t2 = 1 - 2*(qz*qz)

    Wz = np.arctan2(t1,t2)

    return Wz


class Navigation_Metrics(Node):
    def __init__(self, params, logdir):
        """
        """
        super().__init__('pic4rl_nav_metrics')

        self.params = params
        self.previous_time      = time.time()
        self.metrics_results    = []
        self.path               = []

        self.open_logdir(logdir)
        self.init_get_entity_client()

    def init_get_entity_client(self):
        """
        """
        self.robot_pose_client = self.create_client(
            GetEntityState, 
            '/test/get_entity_state'
            )

        while not self.robot_pose_client.wait_for_service(timeout_sec=1):
            self.get_logger().warn('service not available, waiting again...')
        self.req = GetEntityState.Request()
        self.req.name = self.params['robot_name']

        self.get_logger().info("GetEntityState Client online!")

    def get_metrics_data(self, lidar_measurement, episode_step, done=False):
        """
        """
        step_time = time.time()
        robot_pose, velocity = self.acquire_data()

        if episode_step == 0:
            self.start_time         = time.time()
            self.path               = []
            self.goal_path          = []
            self.velocities         = []
            self.accelerations      = []
            self.lidar_measurements = []
            self.metrics_results    = []

            velocity = [0., 0.]
            self.previous_velocity = [0., 0.]
        
        time_delay = step_time-self.previous_time
        acceleration = [
            (velocity[0] - self.previous_velocity[0])/time_delay,
            (velocity[1] - self.previous_velocity[1])/time_delay
            ]

        self.path.append(robot_pose)
        self.velocities.append(velocity)
        self.accelerations.append(acceleration)
        self.lidar_measurements.append(lidar_measurement)

        self.previous_velocity = velocity
        self.previous_time = step_time

    def get_following_metrics_data(self, goal_pose):
        """
        """
        self.goal_path.append(goal_pose)

    def acquire_data(self):
        """
        """
        data = self.get_entity_request()
        robot_pose = [
            data.state.pose.position.x,
            data.state.pose.position.y,
            quat_to_euler(
                data.state.pose.orientation.z,
                data.state.pose.orientation.w
                )
            ]

        velocity = [
            np.hypot(
                data.state.twist.linear.x,
                data.state.twist.linear.y
                ),
            data.state.twist.angular.z
            ]

        return robot_pose, velocity

    def get_entity_request(self):
        """
        """
        future = self.robot_pose_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def create_metric(self, name, value):
        """
        """
        return {'name': name, 'value': value}

    def open_logdir(self, logdir):
        """
        """
        logging.basicConfig(
            filename=os.path.join(logdir, 'metrics_results.log'), 
            level=logging.INFO)

        self.yaml_file = open(logdir + 'metrics_results.yaml', 'w')

    def calc_metrics(
            self, 
            episode=0, 
            start_pose=[0., 0., 0.], 
            goal_pose=[0., 0.],
            true_path=[0., 0.], 
            event='Goal'):
        """
        """
        if not self.path:
            return
            
        if self.params['path_distance']:
            self.path_distance()
        if self.params['distance_path_ratio']:
            self.distance_path_ratio(start_pose, goal_pose)
        if self.params['clearance_time']:
            self.clearance_time()
        if self.params['mean_velocities']:
            self.mean_velocities()
        if self.params['max_min_accelerations']:
            self.max_min_accelerations()
        if self.params['cumulative_heading_average']:
            self.cumulative_heading_average(goal_pose)
        if self.params['following_heading_metrics']:
            self.following_heading_metrics()
        if self.params['obstacle_clereance']:
            self.obstacle_clereance()
        if self.params['row_crop_path_comparison']:
            self.row_crop_path_comparison(true_path)

    def log_metrics_results(self, episode):
        """
        """
        self.get_logger().info(f"Episode {episode+1} metrics results: ")
        logging.info(f"Episode {episode+1} metrics results: ")
        for metric in self.metrics_results:
            self.get_logger().info(f"{metric['name']}: {metric['value']}")
            logging.info(f"{metric['name']}: {metric['value']}")

    def save_metrics_results(self, episode):
        """
        """
        episode_results = {
            f"episode{episode+1}":
            dict(
                (
                    metric['name'],
                    metric['value']
                    )
                for metric in self.metrics_results
                )
            }

        yaml.dump(episode_results, self.yaml_file, default_flow_style=False)
   

    # Metrics
    def path_distance(self, event='Goal'):
        """
        """
        shifted_path = np.delete(self.path, 0, axis=0)
        original_path = np.delete(self.path, -1, axis=0)
        path_distance = np.sum(
            np.hypot(
                original_path[:,0]-shifted_path[:,0], 
                original_path[:,1]-shifted_path[:,1]
                )
            )

        self.metrics_results.append(
            self.create_metric(
                'Path_distance', 
                float(path_distance)
                )
            )

    def distance_path_ratio(self, start_pose, goal_pose, event='Goal'):
        """
        """
        distance = np.hypot(
            start_pose[0]-goal_pose[0], 
            start_pose[1]-goal_pose[1]
            )

        shifted_path = np.delete(self.path, 0, axis=0)
        original_path = np.delete(self.path, -1, axis=0)
        path_distance = np.sum(
            np.hypot(
                original_path[:,0]-shifted_path[:,0], 
                original_path[:,1]-shifted_path[:,1]
                )
            )

        self.metrics_results.append(
            self.create_metric('Distance',  float(distance))
            )
        self.metrics_results.append(
            self.create_metric('Path_distance', float(path_distance))
            )
        self.metrics_results.append(
            self.create_metric(
                'Distance_path_ratio', 
                float(distance/path_distance)
                )
            )

    def clearance_time(self, event='Goal'):
        """
        Time to complete the navigation
        """
        self.metrics_results.append(
            self.create_metric(
                'Clearance_time', 
                float(self.previous_time-self.start_time)
                )
            )

    def mean_velocities(self, event='Goal'): 
        """
        """
        velocities = np.array(self.velocities)
        v, om = velocities[:, 0], velocities[:, 1]

        self.metrics_results.append(
            self.create_metric(
                'Mean_velocities', 
                [float(np.mean(v)), float(np.mean(om))]
                )
            )
        self.metrics_results.append(
            self.create_metric(
                'Std_velocities', 
                [float(np.std(v)), float(np.std(om))]
                )
            )

    def max_min_accelerations(self, event='Goal'):
        """
        """
        accelerations = np.array(self.accelerations)
        linear_acc, yaw_acc = accelerations[:, 0], accelerations[:, 1]

        self.metrics_results.append(
            self.create_metric(
                'Max_linear_acc, Min_linear_acc, Max_yaw_acc, Min_yaw_acc',[
                    float(np.max(linear_acc)), 
                    float(np.min(linear_acc)), 
                    float(np.max(yaw_acc)), 
                    float(np.min(yaw_acc))
                    ]
                )
            )
        self.metrics_results.append(
            self.create_metric(
                'Mean_linear_acc, Mean_yaw_acc',[
                    float(np.mean(linear_acc)), 
                    float(np.mean(yaw_acc))
                    ]
                )
            )

    def cumulative_heading_average(self, goal_pose, event='Goal'):
        """
        """
        delta_thetas = []

        for i in range(len(self.path)):
            _x, _y, _, _ = tf_compose(self.path[i], goal_pose)
            delta_thetas.append(np.arctan2(_y, _x))

        self.metrics_results.append(
            self.create_metric(
                'Cumulative_heading_average', 
                float(np.mean(delta_thetas))
                )
            )
    
    def obstacle_clereance(self, event='Goal'):
        """
        """
        min_distances   = np.min(self.lidar_measurements, axis=1)
        mean_distances  = np.mean(self.lidar_measurements, axis=1)

        self.metrics_results.append(
            self.create_metric(
                'Obstacles_mean_distance', 
                float(np.mean(mean_distances))
                )
            )
        self.metrics_results.append(
            self.create_metric(
                'Obstacles_min_distance', 
                float(np.min(min_distances))
                )
            )
        self.metrics_results.append(
            self.create_metric(
                'Obstacles_min_mean_distance', 
                float(np.min(mean_distances))
                )
            )
        
    def following_heading_metrics(self, event='Goal'):
        """
        """
        if len(self.path) != len(self.goal_path):
            self.get_logger().warn(
                f"Robot path and Goal path have different dimensions ({len(self.path)} and {len(self.goal_path)})"
                )
            return

        delta_thetas = []
        for i in range(len(self.path)):
            _x, _y, _, _ = tf_compose(self.path[i], self.goal_path[i])
            delta_thetas.append(np.arctan2(_y, _x))

        mean        = float(np.mean(delta_thetas))
        mae         = float(np.mean(np.absolute(delta_thetas)))
        rmse        = float(np.sqrt(np.mean(np.array(delta_thetas)**2)))
        max_value   = float(np.max(delta_thetas))

        self.metrics_results.append(
            self.create_metric(
                'Following_heading_metrics', 
                [mean, mae, rmse, max_value]
                )
            )

    def row_crop_path_comparison(self, true_path):
        """
        """
        #true_path = np.array(true_path)
        true_path = true_path*np.ones(len(self.path))
        path = np.array(self.path)

        if len(true_path) != len(path):
            self.get_logger().warn(
                f"Robot tracked path and ground truth path have different dimensions ({len(path)} and {len(true_path)})"
                )
            return

        deviation = true_path - path[:,1]

        mae = float(np.mean(np.absolute(deviation)))
        mse = np.mean(deviation**2)

        self.metrics_results.append(
            self.create_metric(
                'row_crop_path_MAE', 
                float(mae)
                )
            )
        self.metrics_results.append(
            self.create_metric(
                'row_crop_path_MSE', 
                float(mse)
                )
            )



