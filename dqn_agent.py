import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from config import Config

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.config.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.config.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.config.epsilon:
            return random.randrange(self.config.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.config.batch_size:
            return
        minibatch = random.sample(self.memory, self.config.batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay
