from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize Replay Buffer with deque-based storage.
        
        Args:
            capacity: Maximum number of transitions to store (default: 10000)
        """
        self.buffer = deque(maxlen=capacity)
    
    
