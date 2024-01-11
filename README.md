> Project: PIC4 Reinforcement Learning Gym (PIC4rl_gym)

> Owner: "Mauro Martini, Andrea Eirale, Simone Cerrato" 

> Date: "2021:12" 

---

# PIC4 Reinforcement Learning Gym (PIC4rl_gym)

## Description of the project
The PIC4rl_gym project is intended to develop a set of ROS2 packages in order to easily train deep reinforcement learning algorithms for autonomous navigation in a Gazebo simulation environment. A great variety of different sensors (cameras, LiDARs, etc) and platforms are available for custom simulations in both indoor and outdoor stages. Official paper of the project: https://ieeexplore.ieee.org/abstract/document/10193996. Please consider citing our research if you find it useful for your work, google scholar reference at the bottom.

The repository is organized as follows:
- **pic4rl** contains all the ROS 2 packages that handles the communication between the DRL trainer and Gazebo simulation. The package presentis organized in tasks, one for each specific navigation problem. A training class and an environment constitutes the code that define the task together with common utils and tools to plot rewards and organize metrics file. Testing is also possible with the same package, simply enabling the right parameters in the config file. Configuring the training and testing process is possible through two parameters file: main_params.yaml contains all the ROS related configurations, whilst training_params.yaml handles the DRL library. Moreover, in the goal_and_poses folder you can set the list of starting poses and goals for the robot simualation.
- **gazebo_sim** contains simple scripts, models and worlds to start up simulation environments in Gazebo. This package can be substituted by any other launch packages for Gazebo robot simulation, if necessary in your project.

**robot platforms**: the PIC4rl-gym is intended to provide a flexible configurable Gazebo simulation for your training. You can use whatever robotic platform that you have in a ROS 2 / Gazebo package. If you would like to start your work with a set of ready-to-go platforms you can download and add to your workspace the repo PIC4rl_gym_Platforms: https://github.com/PIC4SeR/PIC4rl_gym_Platforms.

You should create your models and worlds for the Gazebo simulation and the respective folders. You can download a full set of worlds and models for Gazebo if you want to use our work for your research:
- worlds download link: https://naspic4ser.polito.it/files/sharing/nehiOIHkY
- models download link: https://naspic4ser.polito.it/files/sharing/9EFOxow8b
(Ask us the password by email: mauro.martini@polito.it, andrea.eirale@polito.it)

## The PIC4rl_gym packages for training agents in simulation:
![alt text](/images/gym_diagram.png "PIC4rl_gym")

## User Guide

**Main scripts in pic4rl training package:**
- **pic4rl_task.py** (inherits environment class: task specific, select and define the agent, define action and state spaces. It is used for both training and testing: load weights of previously trained policy agents)
- **pic4rl_environment_task.py** (specific for the task: interact with the agent with the 'step()' and 'reset()' functions, compute observation and rewards, publish goal, check end of episode condition, call Gazebo services to respawn robot)
- sensors.py: start the necessary sensors topic subscription and contain all the necessary method to preprocess sensor data
- plot_reward.py: utils to plot reward trends for training and validation obtained in a simulation.
- nav_metrics.py: contain all the metrics computed during testing
- evaluate_controller.py: evaluate a generic controller in the same testing framework for metrics comparison and algorithms benchmarking
- evaluate_navigation.py: evaluate a navigation task using the Nav2 framework (simple commander) for benchmarking (Nav2 packages required)

**Config files:**
- main_param.yaml (simulation, sensors, topics, policy selection)
- training_params.yaml (rl training settings)

**COMMANDS:**
- **terminal 1: launch gazebo simulation**
ros2 launch gazebo_sim simulation.launch.py
- **terminal 2: start trainer**
ros2 launch pic4rl pic4rl_starter.launch.py

After a training, we can plot the reward evolution we need to edit the script pic4rl/utils/plot_reward.py and write down the path to the directory of the training (father_path). Then:
- ros2 run pic4rl plot_reward.py

To run the tester, we must modify the param file in pic4rl/config/training_params.yaml. In particular, uncomment the parameter "evaluate" and in "model-dir" write the path to the proper model directory where the checkpoints have been saved. You can copy promising models in the folder pic4rl/models for simplicity. Launch simulation and then the starter with the same terminal commands (colcon-build the workspace if needed).


<img src="/images/Indoor_scenario.png" width="45%" height="45%">

**TO DO**
In the .bashrc export the gazebo models path:

- _export GAZEBO\_MODEL\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim/models _
- _export GAZEBO\_RESOURCE\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim/worlds _

**Tested software versions**
- ROS 2: Foxy (Ubuntu 20.04), Humble (Ubuntu 22.04)
- TensorFlow 2.6.* - 2.10.*
- Keras 2.6.x - 2.10.*

We strongly suggest to set up your learning environment in a docker container starting from pre built cuda images.
**Tested docker images versions**
- nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 (for ROS 2 Foxy)
- nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 (for ROS 2 Humble)

**Try to build tf2rl setup.py:**
- go in the directory: ~/PIC4rl\_gym/training/tf2rl
- pip install .

or install manually the packages in setup.py at ~/PIC4rl\_gym/training/tf2rl/setup.py

## References
    @inproceedings{martini2023pic4rl,
      title={Pic4rl-gym: a ros2 modular framework for robots autonomous navigation with deep reinforcement learning},
      author={Martini, Mauro and Eirale, Andrea and Cerrato, Simone and Chiaberge, Marcello},
      booktitle={2023 3rd International Conference on Computer, Control and Robotics (ICCCR)},
      pages={198--202},
      year={2023},
      organization={IEEE}
    }



