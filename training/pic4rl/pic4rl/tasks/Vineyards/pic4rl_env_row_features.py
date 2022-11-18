#!/usr/bin/env python3

import os
import tensorflow as tf

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
import numpy as np
import math

from numpy import savetxt
import warnings
import cv2
from cv_bridge import CvBridge
from pic4rl.networks.MobileNet_models import Backbone

from rclpy.qos import QoSProfile

class Pic4rlEnvironment(Node):
	def __init__(self):
		super().__init__('pic4rl_environment')
		# To see debug logs
		#rclpy.logging.set_logger_level('pic4rl_environment', 10)

		"""************************************************************
		** Initialise ROS publishers and subscribers
		************************************************************"""
		qos = QoSProfile(depth=10)

		self.cmd_vel_pub = self.create_publisher(
			Twist,
			'cmd_vel',
			qos)

		self.depth_sub = self.create_subscription(
			Image,
					'/camera/depth/image_raw',
					self.depth_callback,
					qos_profile=qos_profile_sensor_data)

		# self.rgb_sub = self.create_subscription(
		# 	Image,
		# 	'/camera/image_raw',
		# 	self.rgb_callback,
		# 	qos_profile=qos_profile_sensor_data)

		self.pause_physics_client = self.create_client(Empty, 'pause_physics')
		self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

		self.get_state_client = self.create_client(State, 'get_state')
		self.new_episode_client = self.create_client(Reset, 'new_episode')

		"""##########
		State variables
		##########"""
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
		self.width = 224
		self.height = 224
		self.show_img = False
		self.bridge = CvBridge()		
		#test variable
		self.step_flag = False
		self.twist_received = None
		self.episode = 0
		# self.image_count = 0
		# self.frame_interval = 12
		# self.num_frame = 1

		# self.depth_image_collection = deque(maxlen = self.frame_interval)
		# self.depth_stack = deque(maxlen = self.num_frame-1)
		self.backbone = Backbone(height = 224, width  = 224,   name = 'backbone')
		self.backbone.model().summary()
		"""##########
		Environment initialization
		##########"""
		
	"""#############
	Main functions
	#############"""

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
		data_received = False
		while not data_received:
			# Send action
			self.send_action(twist)
			# Get state
			state = self.get_state()
			data_received = state.data_received

		lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image = self.process_state(state)

		# Check events (failure,timeout, success)
		done, event = self.check_events(lidar_measurements, goal_distance, self.episode_step)

		if not reset_step:
			# Get reward
			reward = self.get_reward(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)
			observation = self.get_observation(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image)
		else:
			reward = None
			observation = None

		# Send observation and reward
		self.update_state(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)

		return  observation, reward, done

	def reset(self, n_episode):
		self.episode = n_episode
		req = Reset.Request()
		req.goal_pos_x,req.goal_pos_y = self.get_goal()
		self.get_logger().debug("Environment reset ...")

		while not self.new_episode_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		future = self.new_episode_client.call_async(req)
		self.get_logger().debug("Reset env request sent ...")
		#rclpy.spin_until_future_complete(self, future,timeout_sec=1.0)
		#time_start = time.time()
		while rclpy.ok():
			rclpy.spin_once(self,timeout_sec=3)
			if future.done():
				if future.result() is not None:
					self.get_logger().debug("Environment reset done")
					break 
		self.get_logger().debug("Performing null step to reset variables")
		_,_,_, = self._step(reset_step = True)
		observation,_,_, = self._step()
		return observation

	"""#############
	Secondary functions (used in main functions)
	#############"""

	def send_action(self,twist):
		#self.get_logger().debug("unpausing...")
		self.unpause()
		#self.get_logger().debug("publishing twist...")
		self.cmd_vel_pub.publish(twist)
		time.sleep(0.1)
		#self.get_logger().debug("pausing...")
		self.pause()	

	def get_state(self):
		self.get_logger().debug("Asking for the state...")
		req = State.Request()
		future =self.get_state_client.call_async(req)
		rclpy.spin_until_future_complete(self, future)
		try:
			state = future.result()
		except Exception as e:
			node.get_logger().error('Service call failed %r' % (e,))
		self.get_logger().debug("State received ...")
		return state

	def process_state(self,state):
		#from 359 filtered lidar points to 60 selected lidar points
		lidar_measurements = self.process_laserscan(state.scan)

		#from Odometry msg to x,y, yaw, distance, angle wrt goal
		goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_odom(state.odom)

		#process Depth Image from sensor msg
		depth_image = self.depth_image_raw
		depth_image = self.process_depth_image(depth_image)

		return lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image

	def check_events(self, lidar_measurements, goal_distance, step):

		min_range = 0.50

		if  0.05 <min(lidar_measurements) < min_range:
			# Collision
			self.get_logger().info('Collision')
			return True, "collision"

		if goal_distance < 0.50:
			# Goal reached
			self.get_logger().info('Goal')
			return True, "goal"

		if step >= 600:
			#Timeout
			self.get_logger().info('Timeout')
			return True, "timeout"

		return False, "None"

	def get_observation(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw,depth_image):
		
		# WITH CAMERA 
		state_list = []
		# #divide by self.cutoff to normalize
		# # divide by math.pi to normalize goal angle
		#goal_angle_norm = goal_angle
		#goal_info = np.array([goal_distance, goal_angle_norm], dtype=np.float32)
		#goal_info = tf.convert_to_tensor(goal_info)
		#state_list.append(float(goal_distance))
		#state_list.append(float(goal_angle))
		features = np.squeeze(self.get_features(depth_image))
		#state = np.concatenate((goal_info,features))
		state = features
		return state

		# WITH LIDAR
		# state_list = []
		# state_list.append(float(goal_distance))
		# state_list.append(float(goal_angle))
		# for point in lidar_measurements:
		# 	state_list.append(float(point))
		# state = np.array(state_list,dtype = np.float32)
		# #state = np.array([goal_distance, goal_angle, lidar_measurements], dtype = np.float32)
		# return state

	def get_reward(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
		yaw_reward = (0.5 - 2*math.sqrt(math.fabs(goal_angle / math.pi)))
		y_reward = (-2**(math.fabs(4.5 - pos_y))+1)*10
		#distance_reward = 2*((2 * self.previous_goal_distance) / \
		#	(self.previous_goal_distance + goal_distance) - 1)
		#distance_reward = (2 - 2**(self.goal_distance / self.init_goal_distance))
		#distance_reward = (self.previous_goal_distance - goal_distance)*35
		#v = twist.linear.x
		#w = twist.angular.z
		#speed_re = (3*v - math.fabs(w))
        
		reward =  yaw_reward + y_reward 

		if event == "goal":
			reward += 500
		if event == "collision":
			#reward += -1000*math.fabs(v)**2
			reward = -500

		self.get_logger().debug(str(reward))

		# print(
		# 	"Reward:", reward,
		#  	"Yaw r:", yaw_reward,
		# 	"Distance r:", y_reward)
		return reward

	def get_goal(self):
		if self.episode <= 200:
			x = 8.0
			y = 4.5
		elif self.episode > 200 and self.episode <=400:
			x = 12.0
			y = 4.5
		elif self.episode > 400:
			x = 16.0
			y = 4.5
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

	def process_laserscan(self,laserscan_msg):
		# inf or nan values are corrected and they will not be passed to the DRL agent
		scan_range = np.nan_to_num(laserscan_msg.ranges[:], nan=0.0, posinf=self.cutoff, neginf=0.0)
		scan_range_process = []
		min_dist_point = 100
		# Takes only 36 lidar points
		div = 360/self.lidar_points
		for i in range(359):
			if scan_range[i] < min_dist_point:
				min_dist_point = scan_range[i]
			if i % div == 0:
				scan_range_process.append(min_dist_point)
				min_dist_point = 100
		#print('selected lidar points:', len(scan_range_process))

		self.min_obstacle_distance = min(scan_range_process)
		self.max_obstacle_distance = max(scan_range_process)
		self.min_obstacle_angle = np.argmin(scan_range_process)
		#print('min obstacle distance: ', self.min_obstacle_distance)
		#print('min obstacle angle :', self.min_obstacle_angle)
		#print('60 lidar points: ', scan_range_process)

		return scan_range_process
 
	def depth_callback(self, msg):
		depth_image_raw = np.zeros((480,640), np.uint8)
		depth_image_raw = self.bridge.imgmsg_to_cv2(msg, '32FC1')
		self.depth_image_raw = np.array(depth_image_raw, dtype= np.float32)
		#print(self.depth_image_raw.shape)
		#savetxt('/home/mauromartini/mauro_ws/depth_images/text_depth_image_raw.csv', depth_image_raw, delimiter=',')
		#np.save('/home/maurom/depth_images/depth_image.npy', depth_image_raw)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_raw.png', self.depth_image_raw)
	
	def process_depth_image(self,frame):
		np.seterr(all='raise')
		# IF SIMULATION
		max_depth = self.cutoff # [m]
		# IF REAL CAMERA
		#max_depth = self.cutoff*1000 [mm]

		depth_frame = np.nan_to_num(frame, nan=0.0, posinf=max_depth, neginf=0.0)
		depth_frame = np.minimum(depth_frame, max_depth) # [m] in simulation, [mm] with real camera

		depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation = cv2.INTER_AREA)
		depth_frame = np.array(depth_frame, dtype=np.float64)
		depth_frame = depth_frame/max_depth
		
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			try: 
				depth_frame = depth_frame*255.0
				depth_frame = depth_frame.astype(dtype=np.float32)
			except Warning: print ('Raised!')

		if self.show_img:
			self.show_image(depth_frame)
		depth_frame = np.expand_dims(depth_frame, axis = -1)
		depth_frame = np.expand_dims(depth_frame, axis = 0)

		# if 3 channels are needed by the backbone
		depth_frame = np.tile(depth_frame, (1, 1, 3))
		#print(depth_frame.shape)
		return depth_frame

	@tf.function
	def get_features(self, image):
		features = self.backbone(image)
		return features

	def rgb_callback(self, msg):
		rgb_image_raw = np.zeros((480,640,3), np.uint8)
		rgb_image_raw = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
		self.rgb_image_raw = np.array(rgb_image_raw, dtype= np.float32)

		# if show_image:
		processed_img = self.process_rgb_image(self.rgb_image_raw)
		#self.show_image(processed_img, depth=False)

	@tf.function
	def process_rgb_image(self, img):
		#print('image shape: ', img.shape)
		img = tf.reshape(img, [480,640,3])
		img_resize = tf.image.resize(img,[224,224])
		image = tf.expand_dims(img_resize, axis=0)
		return image

	def show_image(self, image):
		colormap = np.asarray(image, dtype = np.uint8)
		cv2.namedWindow('Depth Image', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('Depth Image',colormap)
		cv2.waitKey(1)

	def process_odom(self, odom_msg):
		#self.previous_pose.pose.pose.position.x = odom_msg.pose.pose.position.x
		#self.previous_pose.pose.pose.position.y = odom_msg.pose.pose.position.y

		pos_x = odom_msg.pose.pose.position.x
		pos_y = odom_msg.pose.pose.position.y
		_,_,yaw = self.euler_from_quaternion(odom_msg.pose.pose.orientation)

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



def main(args=None):
	rclpy.init()
	pic4rl_environment = Pic4rlEnvironment()
	pic4rl_environment.spin()

	pic4rl_environment.get_logger().info('Node spinning ...')
	rclpy.spin_once(pic4rl_environment)

	pic4rl_environment.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
