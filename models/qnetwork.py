import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize Q-Network.
        
        Args:
            state_size: Dimension of input state (4 for CartPole)
            action_size: Number of possible actions (2 for CartPole)
            hidden_size: Number of units in hidden layers (default: 128)
        """
        super(QNetwork, self).__init__()
        
        # Define two hidden layers with 128 units each
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Define output layer with action_size units
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for all actions (no final activation)
        """
        # Apply ReLU activation after first hidden layer
        x = F.relu(self.fc1(x))
        
        # Apply ReLU activation after second hidden layer
        x = F.relu(self.fc2(x))
        
        # Return Q-values without final activation
        return self.fc3(x)

    def save(self, filepath):
         """
    Save the model's state dictionary to a file.
    
    Args:
        filepath: Path where the model will be saved
    """
        torch.save(self.state_dict(),filepath)
    
    def load(self,filepath):
        """
    Load the model's state dictionary from a file.
    
    Args:
        filepath: Path to the saved model file
    """
        self.load_state_dict(torch.load(filepath))
        self.eval()