import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            target = reward

            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(state)
            target_f[0][action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Job Scheduling Environment
class JobSchedulingEnv:
    def __init__(self, num_servers, num_jobs):
        self.num_servers = num_servers
        self.num_jobs = num_jobs
        self.state_size = num_servers + num_jobs
        self.action_size = num_servers * num_jobs
        self.reset()

    def reset(self):
        self.servers = np.zeros(self.num_servers)  # servers' current loads
        self.jobs = np.random.randint(1, 10, self.num_jobs)  # jobs' processing times
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.servers, self.jobs])

    def step(self, action):
        server_idx = action // self.num_jobs
        job_idx = action % self.num_jobs

        job_time = self.jobs[job_idx]
        self.servers[server_idx] += job_time
        reward = -job_time  # reward is negative job time (we want to minimize processing time)
        self.jobs[job_idx] = 0  # job is processed and removed from the queue

        done = np.all(self.jobs == 0)
        return self._get_state(), reward, done

# Hyperparameters
num_servers = 3
num_jobs = 5
episodes = 5
batch_size = 32

# Initialize environment and agent
env = JobSchedulingEnv(num_servers, num_jobs)
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
print("input = ",env.state_size)

# Training the agent
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else 10  # larger reward if the job scheduling is done
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            # print(f"Episode: {e+1}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2}")
            break
        agent.replay(batch_size)

# Testing the trained agent
state = env.reset()
state = np.reshape(state, [1, env.state_size])
for time in range(500):
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    # print(f"Action taken: Server {action // num_jobs} -> Job {action % num_jobs}")
    next_state = np.reshape(next_state, [1, env.state_size])
    state = next_state
    if done:
        print("All jobs scheduled.")
        break
