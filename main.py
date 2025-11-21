import gymnasium as gym
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from training.trainer import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN agent on CartPole')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--watch', action='store_true', help='Watch trained agent after training')
    return parser.parse_args()

def watch_agent(agent, env, num_episodes=5):
    """
    Watch the trained agent play.
    
    Args:
        agent: Trained DQNAgent
        env: Gym environment with render_mode='human'
        num_episodes: Number of episodes to watch
    """
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            env.render()
            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()

def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    env = gym.make('CartPole-v1')
    
    agent = DQNAgent(
        state_size=4, 
        action_size=2,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay
    )

    tracker = train(agent, env, num_episodes=args.num_episodes)
    
    print("\nTraining completed!")
    stats = tracker.get_stats()
    print(f"Total episodes: {stats['episodes']}")
    print(f"Final average reward: {stats['average_reward']:.2f}")
    
    if tracker.is_solved():
        print("Environment was solved!")
    else:
        print("Environment was not solved.")
    
    env.close()
    
    if args.watch:
        print("\nWatching trained agent...")
        render_env = gym.make('CartPole-v1', render_mode='human')
        agent.epsilon = 0.0
        watch_agent(agent, render_env, num_episodes=5)


if __name__ == "__main__":
    main()
