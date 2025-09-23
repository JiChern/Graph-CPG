# Graph-CPG
A framework that can generate synchronous waveforms with arbitrary phase-lags in arbitrary size of coupled-oscillator-system. Training is very simple!


[Ji Chen](mailto:ji.chenuk@gmail.com), Song Chen, Yunhan He,  [Li Fan](mailto:fanli77@zju.edu.cn) and [Chao Xu](mailto:cxu@edu.zju.cn)


## Introduction
This repository includes code implementations of the paper titled "Learning Emergent Synchronization in Coupled Oscillators via Graph Attention and Reinforcement Learning" .

## Coupled oscillators controlled by graph attention mechanism
We reconceptualize the problem of waveform generation in coupled-oscillator systems from the viewpoint of swarm intelligence. Our objective is to enable each unit within the coupled system to learn what it should attend to in order to achieve collective objectives. This approach aligns closely with contemporary research on graph attention mechanisms. Based on our concept, each unit learns a distributed strategy where the input is the decomposition of the global goal from the unit's local perspective, and the output is the attention it allocates to other units. 

Here, we propose the graph-CPG model, a concrete implementation of our macro-level concept within a two-dimensional coupled oscillator system. Our task is to enable a coupled oscillator system to generate corresponding oscillatory modes based on user-specified desired phase inputs. For a system with $N$ units, we define the desired phase vector $\mathbf{x}_{\text{dp}} = \{\theta_i\} \in [0, 2\pi]^N$, where $\theta_i$ represents the desired phase lag between the $i$-th node and the first node. For each node $i$, we define its attention to a neighbor $j$ (where $j \in \mathcal{N}(i)$) as $\alpha_{i,j}$. The system of coupled 2D oscillators is then governed by: 

```math
	\dot{\mathbf{x}}_i = f(\mathbf{x}_i) + \text{clamp}\left[\text{MLP}\left(\frac{1}{K}\sum_{k}\sum_{j \in \mathcal{N}(i)} \alpha_{i,j}^k \Theta_t \mathbf{x}_j\right), -1, 1\right],  
	\label{eq:graph-cpg-main}  
```
where $z_i$ represents the state vector of a neuron, $F(z_i)$ is the internal oscillator of the neuron, $R(\theta_i)$ is a 2D rotation matrix with the desired phase lag $\theta_i$, and $\gamma_i$ indicates the coupling strength. The function $\text{Perp}_{z_i}$ is introduced to optimize the transient dynamics. We have uploaded a MATLAB version of the model, which can be executed to observe its performance in gait transition.

<p align="center">
  <img src="https://github.com/JiChern/CPG/blob/main/fig/gait_transition_curves.png?raw=true" alt="Sublime's custom image"/>
</p>

Other types of CPG models [[2]](#1)[[3]](#1) are also available in the 'cpg_matlab' folder, which can be executed to compare with the proposed model.

## CPG-based locomotion control for hexapod robot
We propose a motion generator that is based on the proposed CPG model. This motion generator is responsible for planning the leg motion trajectories for both the stance and swing stages, while also taking into consideration the stability criteria. The overall control framework can be seen in the figure below.

<p align="center">
  <img src="https://github.com/JiChern/CPG/blob/main/fig/motion_fram.jpg?raw=true" alt="Sublime's custom image"/>
</p>

The Python implementation of the motion generation framework has been uploaded in the 'hex_motion_gen' folder. The package allows users to generate limb motion trajectories for further evaluations. It is recommended to evaluate the generated joint trajectories in the Adams simulator.


### Gait transition from caterpillar to metachronal
<p align="center">
  <img src="https://github.com/JiChern/CPG/blob/main/fig/cater_metach.gif?raw=true" alt="Sublime's custom image"/>
</p>

### Gait transition from tetrapod to caterpillar
<p align="center">
  <img src="https://github.com/JiChern/CPG/blob/main/fig/tetra_cater.gif?raw=true" alt="Sublime's custom image"/>
</p>

# Installation
Prerequisites: Ubuntu 20.04, ROS Noetic with Python 3.8, and Gazebo installed.
## create a workspace:
```console
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```
## install the packages:
Clone the hex_motion_gen directory, copy the folders in src directory into the src folder in your workspace. After that, run
```console
$ cd ~/catkin_ws/
$ catkin_make
```
## source the workspace:
```console
$ source ~/catkin_ws/devel/setup.bash
```

# How to use the packages
## Launch the robot model and controllers in Gazebo simulator:
```console
$ roslaunch hexapod gazebo.launch
```
## Run the gait generator
```console
$ rosrun hexapod_control gait_generator.py
```
The gait_generator.py is the Python implementation of the proposed Central Pattern Generator (CPG) model. You can set the initial
gait in this script by changing the input of the function 'set_theta' defined in line 406. All available gait configurations 
are defined in lines 315 to 357.

## Run the trajectory generator
Open a new terminal and run
```console
$ rosrun hexapod_control execute_trajectory_gen.py
```
The execute_trajectory_gen.py is the Python implementation of the proposed hexapod limb motion generator. The initial gait and target
gait can be set by modifying the definitions of the 'start_gait' and 'target_gait' parameters, which are defined in lines 60 and 61. 
It is important to note that the initial gait of the trajectory generator must match the initial gait setting of the gait generator.

## Get the motor data of the robot
After observing the complete gait transition process of the robot in the Gazebo simulator, you can terminate the trajectory generator 
by pressing ctrl+C. The motion trajectories that have been saved can be located in the 'motor_data' folder. These motor data can be 
utilized in the Adams or other simulator for further evaluations.

## Reset the robot
Execute the 'respawn_model.py' node to restore the robot model in the Gazebo simulation. It is important to remember to terminate this
node after the robot has been respawned.
```console
$ rosrun hexapod_control respawn_model.py
```


## References

<a id="1">[1]</a> 
Haitao Yu and Haibo Gao and Liang Ding and Mantian Li and Zongquan Deng and Guangjun Liu (2016). 
Gait generation with smooth transition using CPG-based locomotion control for hexapod walking robot. 
IEEE Transactions on Industrial Electronics, 63, 5488-5500.

<a id="1">[2]</a> 
Ludovic Righetti and Auke Jan Ijspeert (2008). 
Pattern generators with sensory feedback for the control of quadruped locomotion. 
2008 IEEE International Conference on Robotics and Automation, 819-824.

<a id="1">[3]</a> 
Wei Xiao and Wei Wang (2014). 
Hopf oscillator-based gait transition for a quadruped robot. 
2014 IEEE International Conference on Robotics and Biomimetics, 2074-2079.


