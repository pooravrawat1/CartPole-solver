from collections import deque

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
    

