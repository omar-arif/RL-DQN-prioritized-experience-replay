# Deep Q-Network (DQN) with Prioritized Experience Replay

This project implements a **Deep Q-Network (DQN) from scratch** using **PyTorch** and Gym's `highway-fast-v0` environment. The implementation includes custom neural network architecture and training logic without relying on higher-level reinforcement learning libraries.

## Overview

- **Custom DQN Architecture**: A fully customized neural network built from scratch in PyTorch is used to approximate Q-values. A separate target network is employed for stability.
- **Prioritized Experience Replay**: The agent learns from prioritized samples based on their TD error, improving training efficiency.
- **Epsilon-Greedy Action Selection**: The agent follows an epsilon-greedy policy for balancing exploration and exploitation.
- **Training**: The model trains for 2000 epochs with experience replay, batch updates, and target network synchronization.
- **Evaluation**: After training, the agent’s performance is evaluated by running episodes in the environment.

## Requirements

- Python 3.7+
- PyTorch
- Gym
- highway-env
- NumPy
