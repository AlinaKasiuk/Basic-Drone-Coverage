# Drone Coverage using Deep Reinforcement Learning

This repository contains the implementation of a deep reinforcement learning (DRL) approach for optimizing the visual coverage of high-relevance areas by a camera-equipped Unmanned Aerial Vehicle (UAV). The project addresses the challenge of limited power and the need for the drone to return to a base station for recharging, while maximizing partial coverage over time.

## Project Description

The core idea is to train a UAV to patrol an environment, represented as a grid, to optimize visual coverage. The drone operates with a limited battery life and must learn a patrolling strategy that balances exploration/coverage with timely returns to a base station for recharging. This prevents the drone from running out of power and falling.

## Environment

A custom OpenAI Gym environment has been developed to simulate the drone coverage problem. This environment defines the state space, action space, and reward function for the DRL agent.

### State Space

The environment is a grid of size WxH. The drone's position is represented by its (x, y) grid coordinates and its z-position (height). The battery level is also part of the state.

| Num | Observation | Min | Max |
| --- | --- | --- | --- |
| 0 | Current x-pos | 0 | W-1 |
| 1 | Current y-pos | 0 | H-1 |
| 2 | Current z-pos | 0 | Z-1 |
| 4 | Battery level | 0 | 1 |

### Action Space

The drone can perform discrete actions, including movements in various directions and changes in altitude.

| Num | Action |
| --- | --- |
| 0 | Forward |
| 1 | Backward |
| 2 | Left |
| 3 | Right |
| 4 | Up |
| 5 | Down |
| 6 | Forward&Left |
| 7 | Forward&Right |
| 8 | Backward&Left |
| 9 | Backward&Right |

### Reward Function

The reward function is designed to incentivize both exploration/coverage and timely returns to the base station. It has two main components:

1.  **Field of View (FOV) Coverage:** Rewards are calculated based on the values within the camera's FOV, considering camera characteristics.
2.  **Distance to Base:** Rewards are also influenced by the drone's proximity to the base station, encouraging it to return for recharge.

## Neural Network and Training

The project utilizes a neural network (likely a Convolutional Neural Network, given the `cnn` directory) to learn the optimal policy. The `main_RL.py` script is responsible for the training process, which involves:

-   **Initialization:** Setting up the environment and the DRL agent.
-   **Training Loop:** Iteratively interacting with the environment, collecting rewards, and updating the neural network's weights.
-   **TensorBoard Logging:** The presence of `adding tensorboard logs` in commit messages suggests that TensorBoard is used for visualizing training progress and metrics.

## Usage

To run the training process, execute `main_RL.py`.

```bash
python main_RL.py
```

## Files of Interest

-   `main_RL.py`: Main script for training the reinforcement learning agent.
-   `gym_drone/`: Contains the custom OpenAI Gym environment definition.
-   `cnn/`: Contains the neural network architecture and related code.
-   `constants.py`: Defines various constants used in the environment and training.
-   `create_maps.py`: Used for generating the grid maps for the environment.




