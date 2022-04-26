# My first custom drone environment

### Description:
A camera-equipped UAV can fly over the environment to be monitored to optimize the visual coverage of high-relevance areas. 

The power is limited and the drone starts at a base station (BS) full of power, and it is allowed to travel a fixed amount of moves. 
We use Deep Reinforcement Learning to adopt a patrolling strategy to opimize the PARTIAL coverage throught time and guarantee that the drone returns to the base station for recharge and to prevent the falling down.


### Observation:
     
Let's assume that the map is of grid size WxH. Position of the drone is represented as (grid x index, grid y index), where (0,0) is the top left of the grid ((W-1,H-1) is max value)) z-pos is the hight (Z is maximum flying height);
     
Type: Box(4)

Num |    Observation    |   Min   |    Max
----|-------------------|---------|-----------
0   |    Current x-pos  |    0    |    W-1
1   |    Current y-pos  |    0    |    H-1
2   |    Current z-pos  |    0    |    Z-1
4   |    Battery level  |    0    |     1

### Actions:

Type: Discrete(10)

Num  |  Action
-----|------------------
0    |  Forward
1    |  Backward
2    |  Left
3    |  Right
4    |  Up      
5    |  Down   
6    |  Forward&Left
7    |  Forward&Right 
8    |  Backward&Left
9    |  Backward&Right 

### Reward:

Agent’s utility is deﬁned by the reward function.
As the drone can fly in two modes (exploration and looking for recharge) the reward function can be decomposed into two parts.  
The first part calculates from the values incuded in the FOV (field of view) of the camera taking into account camera caracteristics.
The second part is a function of the distance to the base.
The expected reward should be maximized. 


### Starting State:
We should start on one of the base stations choosen randomly


<img src="https://github.com/AlinaKasiuk/my_drone_env_1/blob/main/examples/map.png" width="800">

### Examples:

<img src="https://github.com/AlinaKasiuk/my_drone_env_1/blob/main/examples/example_2.png" width="400">
<img src="https://github.com/AlinaKasiuk/my_drone_env_1/blob/main/examples/example_1.png" width="400">
