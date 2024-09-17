import random

class Config:
    def __init__(self):
        self.learning_rate = 0.001   # Adjust learning rate here
        self.gamma = 0.95            # Discount factor for future rewards
        self.epsilon = 1.0           # Exploration rate
        self.epsilon_min = 0.01      # Minimum exploration rate
        self.epsilon_decay = 0.995   # Decay rate for exploration
        self.batch_size = 32         # Mini-batch size
        self.obstacle_speed = 5      # Speed of the obstacles (change to make game harder)
        self.state_size = 4          # Add this line
        self.action_size = 2         # Add this line
        self.positive_reward = 1
        self.negative_reward = -100
        self.reward_adjustment_rate = 0.1

    def optimize_rewards(self):
        # Randomly adjust rewards to find better values
        self.positive_reward += random.uniform(-self.reward_adjustment_rate, self.reward_adjustment_rate)
        self.negative_reward += random.uniform(-self.reward_adjustment_rate * 10, self.reward_adjustment_rate * 10)
        
        # Ensure rewards stay within reasonable bounds
        self.positive_reward = max(0.1, min(5, self.positive_reward))
        self.negative_reward = max(-200, min(-50, self.negative_reward))
