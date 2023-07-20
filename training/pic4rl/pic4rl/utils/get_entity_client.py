#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetEntityState

class GetEntityClient(Node):

    def __init__(self):
        super().__init__('get_entity_client')

        self.robot_pose_client = self.create_client(GetEntityState, '/test/get_entity_state')
        
        while not self.robot_pose_client.wait_for_service(timeout_sec=1):
            self.get_logger().info('service not available, waiting again...')
        self.req = GetEntityState.Request()
        self.req.name = "goal_box"

        self.get_logger().info("Goal Client online!")

    def send_request(self):
        """
        """
        future = self.robot_pose_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        return future.result()
