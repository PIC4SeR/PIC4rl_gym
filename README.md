> Project: PIC4 Reinforcement Learning Gym (PIC4rl_gym)

> Owner: "Mauro Martini, Andrea Eirale, Simone Cerrato" 

> Date: "2021:12" 

---

# PIC4 Reinforcement Learning Gym (PIC4rl_gym)

## Description of the project
The PIC4rl_gym project is intended to develop a set of ROS2 packages in order to easily train deep reinforcement learning algorithms for autonomous navigation in a Gazebo simulation environment. A great variety of different sensors (cameras, LiDARs, etc) and platforms are available for custom simulations in both indoor and outdoor stages.

The repository is organized as follows:
- **training** contains all the ROS2 packages that handles the communication with Gazebo and define the task environment. It also contains all the DRL packages dedicated to policy selection and training
- **platforms** groups all the ROS2 packages for platform usage in the simulation environment
- **gazebo_sim** contains whatever related to simulation environments
- **testing** contains the ROS2 package to test agent policy

## The PIC4rl_gym packages for training agents in simulation:
![alt text](/images/gym_diagram.png "PIC4rl_gym")

## User Guide

**Main scripts in pic4rl training package:**
- **pic4rl_trainer.py** (instanciate the agent and start the main training loop)
- **pic4rl_training_.py** (inherits environment class: task specific, select and define the agent, define action and state spaces)
- **pic4rl_environment_.py** (specific for the Task: interact with the agent with the 'step()' and 'reset()' functions, compute observation and rewards, publish goal, check end of episode condition, call Gazebo services to respawn robot)
- sensors.py: start the necessary sensors topic subscription and contain all the necessary method to preprocess sensor data
**Main scripts in pic4rl testing package:**
- **pic4rl_tester.py** (load weights and instanciate policy agents and environment)
- metrics.py: contain all the metrics computed during testing

**ROS Nodes:**
- pic4rl_training(pic4rl_environment(Node))
- pic4rl_testing(pic4rl_environment(Node))

**Config files:**
- main_param.yaml (simulation, sensors, topics, policy selection)
- training_params.yaml (rl training settings)

**COMMANDS:**
- **terminal 1: launch gazebo simulation**
ros2 launch gazebo_sim simulation.launch.py
- **terminal 2: start trainer**
ros2 run pic4rl pic4rl_trainer


<img src="/images/Indoor_scenario.png" width="45%" height="45%">

**TO DO**
In the .bashrc export the gazebo models path:

- _export GAZEBO\_MODEL\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim/models _
- _export GAZEBO\_MODEL\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim _

**Tested software versions**
- ROS2 Foxy
- TensorFlow 2.6.x
- Keras 2.6.x

**Try to build tf2rl setup.py:**
- go in the directory: ~/PIC4rl\_gym/training/tf2rl
- pip install .

or install manually the packages in setup.py at ~/PIC4rl\_gym/training/tf2rl/setup.py

