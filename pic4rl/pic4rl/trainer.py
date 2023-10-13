import os
import yaml
import time
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs import tensor_spec
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy

from pic4rl.tasks.goToPose.env_lidar import Pic4rlEnv_Lidar
from pic4rl.tasks.goToPose.env_camera import Pic4rlEnv_Camera
from pic4rl.tasks.Vineyards.env_vineyard import Pic4rlEnv_Vineyard
from pic4rl.tasks.Following.env_lidar_pf import Pic4rlEnv_Lidar_PF
from pic4rl.utils.train_utils import TrainUtils
import rclpy
from ament_index_python.packages import get_package_share_directory

# Local variables to be defined
train_utils = TrainUtils()

def _get_envs(params, logger):
    """
    """
    if params.task == 'goToPose':
        if params.sensor == 'lidar':
            _Env = Pic4rlEnv_Lidar
        elif params.sensor == 'camera':
            _Env = Pic4rlEnv_Camera
        else:
            raise Exception("Sensor not valid for task goToPose!")
    elif params.task == 'Vineyards':
        _Env = Pic4rlEnv_Vineyard
    elif params.task == 'Following':
        _Env = Pic4rlEnv_Lidar_PF
    else:
        raise Exception('Task not valid')

    _train  = _Env("Main_Env")                   # Used for training
    _eval   = TFPyEnvironment(_Env("Eval_Env"))  # Used for evaluation
    _warmup = _Env("Warmup_Env")                 # Used for warmup

    logger.debug(f"Network layers: {params.layer_units}")
    logger.debug(f"{_train.time_step_spec().observation}\n{_train.action_spec()}\n")

    return _train, _eval, _warmup, _Env

def _get_agent(params, tf_env):
    """
    """
    if params.policy == 'DQN':
        get_agent = train_utils.get_dqn_agent
    elif params.policy == 'DDPG':
        get_agent = train_utils.get_ddpg_agent
    elif params.policy == 'TD3':
        get_agent = train_utils.get_td3_agent
    elif params.policy == 'SAC':
        get_agent = train_utils.get_sac_agent
    else:
        raise SystemExit('Policy not valid')

    observation_spec    = tf_env.observation_spec()
    action_spec         = tf_env.action_spec()
    time_step_spec      = tf_env.time_step_spec()

    _agent = get_agent(params, observation_spec, action_spec, time_step_spec)
    _agent.train = common.function(_agent.train) 
    
    return _agent

def _get_drivers(params,logger,train_env,eval_env,warmup_env,agent,
        replay_buffer,rb_observer):
    """
    """
    if params.retrain or params.evaluate:
        # load weights
        train_utils.load_ckpt(params.model_dir, agent, replay_buffer)
        logger.info(f"\nWeights successfully loaded from {params.model_dir}\n")

        warmup_driver = py_driver.PyDriver(
            warmup_env,
            py_tf_eager_policy.PyTFEagerPolicy(agent.policy, use_tf_function=True),
            [rb_observer],
            max_steps=params.warmup_steps)

    else:
        warmup_policy  = random_tf_policy.RandomTFPolicy(
            eval_env.time_step_spec(), 
            eval_env.action_spec()
            )
        warmup_driver = py_driver.PyDriver(
            warmup_env,
            py_tf_eager_policy.PyTFEagerPolicy(warmup_policy, use_tf_function=True),
            [rb_observer],
            max_steps=params.warmup_steps)


    # Training Drivers
    collect_driver = py_driver.PyDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=params.rb_steps_per_iter
    )

    return warmup_driver, collect_driver

def _train(params,train_env,eval_env,warmup_env,collect_driver,warmup_driver,
        replay_buffer,iterator,agent,results_path,logger):
    """
    """
    # Warmup
    time_step = warmup_env.reset()

    warmup_driver.run(time_step)

    # Training
    time_step = train_env.reset()
    returns = []

    for _ in range(params.max_steps):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % params.eval_interval == 0:
            avg_return = compute_avg_return(
                eval_env, 
                agent.policy, 
                params.eval_episodes
                )
            logger.info(f"Average return on {params.eval_episodes} episodes: {avg_return}\n")
            returns.append(avg_return)

    train_utils.compute_avg_return(eval_env, agent.policy, params.eval_episodes)

    # Save chekpoint and policy
    train_utils.save_ckpt(results_path, agent, replay_buffer)

def main():
    # Parsing Hyperparameters
    rclpy.init()
    params_path = os.path.join(
        get_package_share_directory('pic4rl'), 'config/params.yaml')
    params = train_utils.get_parameters(params_path)  

    # Define directories and logger
    if params.evaluate:
        logger = train_utils.set_logger()
    else:
        results_path, log_path = train_utils.define_path(params)
        logger = train_utils.set_logger(log_path) 

    train_env, eval_env, warmup_env, _Env = _get_envs(params, logger)
    agent = _get_agent(params, eval_env)
    replay_buffer, rb_observer, iterator = train_utils.create_replay_buffer(
        agent, params.rb_size)
    warmup_driver, collect_driver = _get_drivers(params,logger,train_env,
        eval_env,warmup_env,agent,replay_buffer,rb_observer)

    if params.evaluate:
        logger.info(f"\nStarting evaluating for {params.evaluate_episodes} episodes\n")
        avg_return = train_utils.compute_avg_return(eval_env,agent.policy,
            params.evaluate_episodes)
        logger.info(f"Average return on {params.evaluate_episodes} episodes: {avg_return}\n")
    else:
        try:
            _train(params,train_env,eval_env,warmup_env,collect_driver,
                warmup_driver,replay_buffer,iterator,agent,results_path,logger)
        except KeyboardInterrupt:
            train_utils.save_ckpt(results_path, agent, replay_buffer)

if __name__ == '__main__':
    main()