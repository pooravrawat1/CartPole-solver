from utils.performance_tracker import PerformanceTracker

def train(agent, env, num_episodes=500):
    """
    Train the DQN agent in the given environment.
    
    Args:
        agent: DQNAgent instance
        env: Gym environment
        num_episodes: Number of episodes to train (default: 500)
    """
    tracker = PerformanceTracker()
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            
            agent.train_step()
            
            if agent.training_steps % 100 == 0:
                agent.update_target_network()
            
            state = next_state
        
        agent.decay_epsilon()
        tracker.add_episode_reward(episode_reward)
        
        if (episode + 1) % 10 == 0:
            stats = tracker.get_stats()
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {stats['average_reward']:.2f} | "
                  f"Latest: {stats['latest_reward']} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        if tracker.is_solved():
            print(f"\nEnvironment solved in {episode + 1} episodes!")
            print(f"Average reward: {tracker.get_average_reward():.2f}")
            break
    
    return tracker