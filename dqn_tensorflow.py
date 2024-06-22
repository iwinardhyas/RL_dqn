import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network for Deep Q-learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Environment setup for job scheduling
class JobSchedulingEnv:
    def __init__(self, num_servers, num_jobs):
        self.num_servers = num_servers
        self.num_jobs = num_jobs
        self.state_size = num_servers + num_jobs
        self.action_size = num_servers * num_jobs
        self.reset()

    def reset(self):
        # Initialize servers' and jobs' states
        self.servers = np.zeros(self.num_servers)  # servers load
        self.jobs = np.random.randint(1, 10, self.num_jobs)  # jobs processing times
        return self._get_state()

    def _get_state(self):
        # Concatenate server loads and job times
        return np.concatenate([self.servers, self.jobs])

    def step(self, action):
        server_idx = action // self.num_jobs
        job_idx = action % self.num_jobs

        job_time = self.jobs[job_idx]
        self.servers[server_idx] += job_time
        reward = -job_time  # Negative reward for processing time
        self.jobs[job_idx] = 0  # Job is processed

        done = np.all(self.jobs == 0)
        return self._get_state(), reward, done

# Parameters
num_servers = 3
num_jobs = 5
episodes = 500
batch_size = 32

env = JobSchedulingEnv(num_servers, num_jobs)
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

# Training
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else 10
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Test the trained agent
state = env.reset()
state = np.reshape(state, [1, env.state_size])
for time in range(500):
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    print(f"Action taken: Server {action // num_jobs} -> Job {action % num_jobs}")
    next_state = np.reshape(next_state, [1, env.state_size])
    state = next_state
    if done:
        print("All jobs scheduled.")
        break
