#!/usr/bin/env python3

import yaml
import rclpy
import threading
from pic4rl.tasks.goToPose.pic4rl_lidar import Pic4rlTesting_Lidar
from pic4rl.tasks.goToPose.pic4rl_camera import Pic4rlTesting_Camera
from pic4rl.tasks.Vineyards.pic4rl_vineyard import Pic4rlTesting_Vineyard
from pic4rl.tasks.Following.pic4rl_lidar_pf import Pic4rlTesting_Lidar_PF
from ament_index_python.packages import get_package_share_directory
from pic4rl.tasks.following.pic4rl_environment_lidar_pf import GetEntityClient

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

def main(args=None):
    """
    """
    configFilepath = os.path.join(
        get_package_share_directory("testing"), 'config',
        'main_param.yaml'
    )

    with open(configFilepath, 'r') as file:
        configParams = yaml.safe_load(file)['main_node']['ros__parameters']

    rclpy.init()

    if configParams['task']=='following':
        get_entity_client = GetEntityClient()
        
        if configParams['sensor'] == 'lidar':
            pic4rl= Pic4rlTesting_Lidar_PF(get_entity_client)
            pic4rl.get_logger().info(
                "Initialized Testing: sensor=LiDAR, task=Following\n\n")

    elif configParams['task']=='vineyards':

        if configParams['sensor'] == 'camera':
            pic4rl= Pic4rlTesting_Vineyard()
            pic4rl.get_logger().info(
                "Initialized Testing: sensor=Camera, task=P2P\n\n")

    else:

        if configParams['sensor'] == 'lidar':
            pic4rl= Pic4rlTesting_Lidar()
            pic4rl.get_logger().info(
                "Initialized Testing: sensor=LiDAR, task=P2P\n\n")
        elif configParams['sensor'] == 'camera':
            pic4rl= Pic4rlTesting_Camera()
            pic4rl.get_logger().info(
                "Initialized Testing: sensor=Camera, task=P2P\n\n")
    
    pic4rl.threadFunc()

    pic4rl.destroy_node()
    rclpy.shutdown()

    # th = threading.Thread(target=pic4rl.threadFunc)    
    # th.start()
    
    # try:
    #     rclpy.spin(pic4rl)
    # except:
    #     pic4rl.destroy_node()
    #     th.join()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()