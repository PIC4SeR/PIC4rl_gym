> Project: PIC4 Reinforcement Learning Gym (PIC4rl_gym)

> Owner: "Mauro Martini, Andrea Eirale, Simone Cerrato" 

> Date: "2021:12" 

---

# PIC4 Reinforcement Learning Gym (PIC4rl_gym)

## Description of the project
The PIC4rl_gym project is intended to develop a set of ROS2 packages in order to easily train deep reinforcement learning algorithms for autonomous navigation in a Gazebo simulation environment. A great variety of different sensors (cameras, LiDARs, etc) and platforms are available for custom simulations in both indoor and outdoor stages. Official paper of the project: https://ieeexplore.ieee.org/abstract/document/10193996. Please consider citing our research if you find it useful for your work, google scholar reference at the bottom.

The repository is organized as follows:
- **pic4rl** contains all the ROS 2 packages that handles the communication with Gazebo and define the task environment. It also contains all the DRL packages dedicated to policy selection and training. Here you can also find ROS 2 package to test agent policy
- **gazebo_sim** contains whatever related to simulation environments

- The PIC4rl-gym is intended to provide a flexible configurable Gazebo simulation for your training. You can use whatever robotic platform that you have in a ROS 2 / Gazebo package. If you would like to start your work with a set of ready-to-go platforms you can download and add to your workspace the repo PIC4rl_gym_Platforms: https://github.com/PIC4SeR/PIC4rl_gym_Platforms.

You can download a full set of worlds and models for Gazebo if you want to use our work for your research:
- worlds download link: https://naspic4ser.polito.it/files/sharing/nehiOIHkY
- models download link: https://naspic4ser.polito.it/files/sharing/9EFOxow8b
(Ask us the password by email: mauro.martini@polito.it, andrea.eirale@polito.it)

## The PIC4rl_gym packages for training agents in simulation:
![alt text](/images/gym_diagram.png "PIC4rl_gym")

## User Guide

**Main scripts in pic4rl training package:**
- **trainer.py** (instanciates the agent and starts the main training loop, selects and defines the agent)
- **env_.py** (specific for the Task: defines action and state spaces, interacts with the agent with the 'step()' and 'reset()' functions, computes observation and rewards, publishes goal, checks end of episode condition, calls Gazebo services to respawn robot)
- sensors.py: starts the necessary sensors topic subscription and contains all the necessary method to preprocess sensor data

**ROS Nodes:**
- trainer()

**Config files:**
- params.yaml (simulation, sensors, topics, policy selection, rl training settings)

**COMMANDS:**
- **terminal 1: launch gazebo simulation**
ros2 launch gazebo_sim simulation.launch.py
- **terminal 2: start trainer**
ros2 run pic4rl trainer

In the configuration file params.yaml you can specify the directory to contain the training results under the "logdir" voice. Then, with the trained weights, you can both:
- Continue training these weights with the retrain function
- Evaluate the trained weights

In both the cases, you want to copy the path of the trained weights in the "model-dir" voice of params.yaml, then remove the comment to the "retrain" or "evaluate" parameter, whether you want to continue training or evaluate the trained weights. Then run:
- ros2 run pic4rl trainer


<img src="/images/Indoor_scenario.png" width="45%" height="45%">

**TO DO**
In the .bashrc export the gazebo models path:

- _export GAZEBO\_MODEL\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim/models _
- _export GAZEBO\_RESOURCE\_PATH=$GAZEBO\_MODEL\_PATH=:<YOUR_WORKSPACE_DIR>/src/PIC4rl\_gym/simulation/gazebo_sim/worlds _

**Tested software versions**
- ROS 2: Foxy (Ubuntu 20.04), Humble (Ubuntu 22.04)
- TensorFlow 2.10* or newer

We strongly suggest to set up your learning environment in a docker container starting from pre built cuda images.
**Tested docker images versions**
- nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 (for ROS 2 Foxy)
- nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 (for ROS 2 Humble)

**Install TFAgents**
- pip install tf-agents[reverb]

## References
    @inproceedings{martini2023pic4rl,
      title={Pic4rl-gym: a ros2 modular framework for robots autonomous navigation with deep reinforcement learning},
      author={Martini, Mauro and Eirale, Andrea and Cerrato, Simone and Chiaberge, Marcello},
      booktitle={2023 3rd International Conference on Computer, Control and Robotics (ICCCR)},
      pages={198--202},
      year={2023},
      organization={IEEE}
    }



