import torch
import torch.optim as optim
import numpy as np
from models.qnetwork import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, buffer_capacity=10000):
                """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer (default: 0.001)
            gamma: Discount factor for future rewards (default: 0.99)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Decay rate for epsilon (default: 0.995)
            epsilon_min: Minimum epsilon value (default: 0.05)
            buffer_capacity: Replay buffer capacity (default: 10000)
        """
        self.state_size = state_size 
        self.action_size = action_size

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(capacity = buffer_capacity)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr = learning_rate)

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.training_steps = 0

    def select_action(self, state):
        """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current state
        
    Returns:
        Selected action (integer)
    """
    if np.random.random() < self.epsilon:
        # Explore: select random action
        return np.random.randint(self.action_size)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    
    def store_transitions(self, state, action, next_state, done):
            """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        







