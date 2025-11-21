# CartPole DQN Solver

A Deep Q-Network (DQN) implementation that learns to balance a pole on a moving cart using reinforcement learning. Built from scratch with PyTorch.

## Results

- **Solved in 320 episodes** (average reward: 196.88 over last 100 episodes)
- **Perfect performance**: 5/5 test episodes with maximum score (500/500)
- **Training time**: Approximately 5-10 minutes on CPU
- **Success criteria**: Average reward ≥ 195 over 100 consecutive episodes

## Overview

CartPole is a classic reinforcement learning problem where an agent learns to balance a pole on a moving cart by applying forces to push the cart left or right. The agent:
- Begins with no prior knowledge and takes random actions
- Learns from experience through trial and error
- Discovers an optimal strategy to maintain pole balance
- Achieves consistent maximum performance (500 steps per episode)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CartPole-solver.git
cd CartPole-solver

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Train the agent:**
```bash
python main.py
```

**Train and visualize the agent:**
```bash
python main.py --watch
```

**Customize hyperparameters:**
```bash
python main.py --num_episodes 1000 --learning_rate 0.0005 --epsilon_decay 0.99
```

## Training Progress

The agent demonstrates progressive learning:

```
Episode 10/500  | Avg Reward: 29.50  | Epsilon: 0.951
Episode 100/500 | Avg Reward: 39.16  | Epsilon: 0.606
Episode 200/500 | Avg Reward: 67.87  | Epsilon: 0.367
Episode 280/500 | Avg Reward: 109.81 | Epsilon: 0.246  (First maximum score)
Episode 320/500 | Avg Reward: 196.88 | Epsilon: 0.201  (Environment solved)

Environment solved in 320 episodes!
```

## Architecture

### Q-Network
- **Input Layer**: 4 state values (cart position, cart velocity, pole angle, pole angular velocity)
- **Hidden Layers**: 2 fully connected layers with 128 units each (ReLU activation)
- **Output Layer**: 2 Q-values (one for each action: push left or push right)

### DQN Components
- **Experience Replay**: Maintains a buffer of 10,000 past transitions for stable learning
- **Target Network**: Updated every 100 training steps to provide stable learning targets
- **Epsilon-Greedy Policy**: Balances exploration and exploitation
  - Initial epsilon: 1.0 (100% random exploration)
  - Final epsilon: 0.05 (5% random exploration)
  - Decay rate: 0.995 per episode

### Hyperparameters
- **Learning Rate**: 0.001 (Adam optimizer)
- **Discount Factor (gamma)**: 0.99
- **Batch Size**: 64
- **Replay Buffer Capacity**: 10,000 transitions

## Project Structure

```
CartPole-solver/
├── agents/
│   └── dqn_agent.py          # DQN agent with training logic
├── models/
│   └── qnetwork.py            # Neural network architecture
├── utils/
│   ├── replay_buffer.py       # Experience replay memory
│   └── performance_tracker.py # Training metrics tracker
├── training/
│   └── trainer.py             # Training loop
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Key Concepts

### Deep Q-Learning
Utilizes a neural network to approximate Q-values (expected cumulative rewards) for each action in a given state.

### Experience Replay
Stores past experiences in a replay buffer and samples random batches for training, which breaks temporal correlations between consecutive samples and improves learning stability.

### Target Network
Maintains a separate network for computing target Q-values. This network is updated periodically rather than at every step, providing more stable learning targets.

### Epsilon-Greedy Exploration
Implements a strategy that balances exploration (trying new actions) with exploitation (using learned knowledge). Epsilon decays over time as the agent becomes more confident.

### Bellman Equation
The core update rule for Q-learning:
```
Q(s,a) = reward + gamma * max(Q(s',a'))
```
This equation updates Q-values based on the immediate reward plus the discounted maximum future reward.

## Technologies

- **Python 3.9+**
- **PyTorch**: Deep learning framework for neural network implementation
- **Gymnasium**: OpenAI's toolkit for developing and comparing reinforcement learning algorithms
- **NumPy**: Library for numerical computations

## Future Enhancements

- Add training visualization and performance plots
- Implement Double DQN to reduce overestimation bias
- Integrate Dueling DQN architecture for improved value estimation
- Implement Prioritized Experience Replay for more efficient learning
- Extend implementation to additional environments (MountainCar, LunarLander)
- Add model persistence (save/load trained models)

## Learning Outcomes

This project provided hands-on experience with:
- Deep reinforcement learning fundamentals
- Q-learning and temporal difference methods
- Neural network training and optimization with PyTorch
- Exploration-exploitation trade-offs
- Experience replay buffer implementation
- Target network stabilization techniques
- Hyperparameter tuning and debugging RL agents

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Mnih, V., et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- OpenAI Gymnasium Documentation
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction

---

Developed as a learning project to understand and implement deep reinforcement learning algorithms.
