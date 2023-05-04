#!/usr/bin/env python3

import yaml
import rclpy
import threading
from pic4rl.tasks.goToPose.pic4rl_training_lidar import Pic4rlTraining_Lidar
from pic4rl.tasks.goToPose.pic4rl_training_camera import Pic4rlTraining_Camera
#from pic4rl.tasks.Vineyards.pic4rl_training_depth import Pic4rlTraining_Vineyard
from pic4rl.tasks.Following.pic4rl_training_lidar_pf import Pic4rlTraining_Lidar_PF
from ament_index_python.packages import get_package_share_directory
from pic4rl.tasks.Following.pic4rl_environment_lidar_pf import GetEntityClient

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
        get_package_share_directory("pic4rl"), 'config',
        'main_params.yaml'
    )

    with open(configFilepath, 'r') as file:
        configParams = yaml.safe_load(file)['main_node']['ros__parameters']

    rclpy.init()

    if configParams['task']=='goToPose':
        if configParams['sensor'] == 'lidar':
            pic4rl_training = Pic4rlTraining_Lidar()
            pic4rl_training.get_logger().info(
                "Initialized Training: sensor=LiDAR, task=goToPose\n\n")
        if configParams['sensor'] == 'camera':
            pic4rl_training = Pic4rlTraining_Camera()
            pic4rl_training.get_logger().info(
                "Initialized Training: sensor=Camera, task=goToPose\n\n")

    elif configParams['task']=='Following':
        get_entity_client = GetEntityClient()
        
        if configParams['sensor'] == 'lidar':
            pic4rl_training = Pic4rlTraining_Lidar_PF (get_entity_client)
            pic4rl_training.get_logger().info(
                "Initialized Training: sensor=LiDAR, task=Following\n\n")

    #elif configParams['task']=='Vineyards':
        ## TO DO 
        # if configParams['sensor'] == 'camera':
        #     pic4rl_training = Pic4rlTraining_Vineyards()
        #     pic4rl_training.get_logger().info(
        #         "Initialized Training: sensor=Camera, task=Vineyards\n\n")
    
    pic4rl_training.threadFunc()

    pic4rl_training.destroy_node()
    rclpy.shutdown()

    # th = threading.Thread(target=pic4rl_training.threadFunc)    
    # th.start()
    
    # try:
    #     rclpy.spin(pic4rl_training)
    # except:
    #     pic4rl_training.destroy_node()
    #     th.join()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()