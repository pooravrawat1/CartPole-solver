import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode="human")

def Random_games():
    # Each of this episode is its own game.
    for episode in range(10):
        print(f"Episode {episode + 1}")
        env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        for t in range(500):
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, truncated, and other info.
            next_state, reward, done, truncated, info = env.step(action)
            
            # lets print everything in one line:
            print(t, next_state, reward, done, truncated, info, action)
            if done or truncated:
                break
                
Random_games()