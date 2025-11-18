from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize Replay Buffer with deque-based storage.
        
        Args:
            capacity: Maximum number of transitions to store (default: 10000)
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, sction, reward, next_state, done):
            """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended (True/False)
        """
        self.buffer.append(state, action, reward, next_state, done)

    def sample(self, batch_size):
            """
        Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
    
    def __len__(self):
        """
        Return the current number of transitions in the buffer.
    
        Returns:
         Number of stored transitions
        """   
        return len(self.buffer)
        


