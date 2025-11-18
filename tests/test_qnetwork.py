import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.qnetwork import QNetwork

def test_forward_pass_output_shape():
    state_size = 4
    action_size = 2
    batch_size = 1


    network = QNetwork(state_size, action_size) #create network

    state = torch.randn(batch_size, state_size) #create dummy input state

    # Forward pass
    output = network(state)

    assert output.shape == (batch_size, action_size), \
        f"Expected shape ({batch_size},{action_size}), got {output.shape}"
def test_save_load_preserves_weights():
    """Test that save/load preserves network weights"""
    state_size = 4
    action_size = 2
    filepath = 'test_model.pth'
    
    # Create and save network
    network1 = QNetwork(state_size, action_size)
    network1.save(filepath)
    
    # Create new network and load weights
    network2 = QNetwork(state_size, action_size)
    network2.load(filepath)
    
    # Compare weights
    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        assert torch.equal(param1, param2), "Weights don't match after load"
    
    # Clean up test file
    if os.path.exists(filepath):
        os.remove(filepath)


def test_forward_pass_with_batch():
    """Test forward pass with multiple states (batch)"""
    state_size = 4
    action_size = 2
    batch_size = 32
    
    network = QNetwork(state_size, action_size)
    states = torch.randn(batch_size, state_size)
    
    output = network(states)
    
    assert output.shape == (batch_size, action_size), \
        f"Expected shape ({batch_size}, {action_size}), got {output.shape}"