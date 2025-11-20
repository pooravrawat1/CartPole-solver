import torch
import torch.nn as nn
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

    
    def store_transition(self, state, action, reward, next_state, done):
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

    def train_step(self, batch_size=64):
        """
        Perform one training step on a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample (default: 64)
            
        Returns:
            Loss value, or None if not enough experiences in buffer
        """
        # 1. Check if we have enough experiences to learn from
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 2. Sample a random batch of past experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 3. Compute current Q-values (what we predicted before)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 4. Compute target Q-values (what we should have predicted)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # Bellman equation: reward + gamma * max(next Q-values) * (1 if not done else 0)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 5. Calculate loss (how wrong were we?)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 6. Backpropagation (learn from mistakes)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()
        
        # 7. Track progress
        self.training_steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        """
        Copy weights from Q-Network to Target Network.
        
        This provides stable learning targets by periodically syncing
        the Target Network with the learned Q-Network weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decay epsilon to shift from exploration to exploitation.
    
        Multiplies epsilon by decay factor, but never goes below epsilon_min.
        This gradually reduces random exploration as the agent learns.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)




