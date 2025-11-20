from collections import deque

class PerformanceTracker:
    def __init__(self, window_size=100):
        """
        Initialize Performance Tracker.
        
        Args:
            window_size: Size of rolling window for average calculation (default: 100)
        """
        # Rolling window - keeps only last 100 episodes
        self.reward_window = deque(maxlen=window_size)
        
        # Full history - keeps everything
        self.all_rewards = []
        self.all_losses = []
    
    def add_episode_reward(self, reward):
        """
        Add an episode reward to tracking.
        
        Args:
            reward: Total reward from completed episode
        """
        self.reward_window.append(reward)
        self.all_rewards.append(reward)

    def get_average_reward(self):
        """
        Calculate average reward over the rolling window.
        
        Returns:
            Average reward, or 0.0 if no episodes recorded yet
        """
        if len(self.reward_window) == 0:
            return 0.0
        return sum(self.reward_window) / len(self.reward_window)
    
    def is_solved(self, threshold=195.0):
        if len(self.reward_window) == self.reward_window.maxlen and self.get_average_reward() >= threshold:
            return True
        else:
            return False

    def get_stats(self):
        stats = {}
        stats["episodes"]  = len(self.all_rewards)
        stats["average_reward"] = self.get_average_reward()
        if len(self.all_rewards) > 0:
            stats["latest_reward"] = self.all_rewards[-1]
        else:
            stats["latest_reward"] = 0

        return stats



    
