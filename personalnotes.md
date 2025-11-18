# A markdown file for storing my personal notes for this project. Can be accessed by anyone who is interested in learning this like I am.

# Reinforcement learning - 
A type of machine learning that allows us to make AI agents that learns by the environment it is in by interacting with it in order to maximize its cumulative reward. Basically using trial and error, and incentivizing by rewarding itself for good actions and punishing itself for bad actions.

After each action, the agent recieves feedback. Feedback consists of the reward and next state of the environment.

Reward is defined by human.
![alt text](image.png)


Started with making a Qnetwork class
The QNetwork class is the "brain" of the AI agent. It is a neural network that learns to predict how good each action is in any given situation.

How it works:
Takes in the current state (like the pole's angle and cart position)
Processes it through two hidden layers (the "thinking" part)
Outputs a score for each possible action (left or right)
The agent picks the action with the highest score

Why it matters: Instead of random moves, this network learns from experience which actions lead to keeping the pole balanced. Over time, it gets better at predicting "if I move left now, I'll probably keep the pole up longer" vs "if I move right, the pole will fall."

The forward pass is literally how the brain thinks.
In the simplest terms:

You give it the current situation (pole angle, cart position, etc.) and it spits out two numbers:

Score for moving LEFT
Score for moving RIGHT
The process:

Takes in the game state (4 numbers describing the pole and cart)
Runs it through the first "thinking" layer → applies ReLU (keeps positive signals, zeros out negative)
Runs it through the second "thinking" layer → applies ReLU again
Outputs two final scores
Example:

Input: [cart position: 0.5, cart speed: 0.2, pole angle: 0.1, pole speed: -0.3]
Output: [Left: 2.3, Right: 4.7]
Decision: Pick RIGHT because 4.7 > 2.3

Wrote a replay buffer function in utils/reply_buffer.py which stores past experiments in a deque, so AI can learn from them.

Added a push method to store transitions
A transition is one moment of gameplay captured as: "I was in this state, took this action, got this reward, and ended up in this new state."

Added a sample method for batch retieval. 
Instead of learning from experiences one at a time, the AI learns from a random batch (like 32 experiences at once). This method grabs a random sample of past experiences for training.