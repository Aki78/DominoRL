import random
import numpy as np
import torch
import torch.optim as optim
import gym
from models import QNet

class DQNAgent:
    def __init__(self, state_size, action_size, lr, gamma, epsilon, epsilon_min, epsilon_decay, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def train(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones)).float()

        # Compute the Q-values for the current states
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values for the next states
        next_q_values = self.target_net(next_states)
        next_q_values, _ = torch.max(next_q_values, dim=1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss and update the Q-network
        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update_target_net()

        # Update the exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.gamma * target_param.data + (1 - self.gamma) * param.data)






# Define the main function for training the agent
def train(env, agent, num_episodes, max_steps):
    replay_buffer = []
    scores = []

    for i_episode in range(1, num_episodes+1):
        # Reset the environment and get the initial state
        state = env.reset()
        score = 0

        for t in range(max_steps):
            # Choose an action using the DQN policy
            action = agent.act(state)

            # Take a step in the environment and store the experience in the replay buffer
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))

            # Update the score and the state
            score += reward
            state = next_state

            # Train the DQN agent
            agent.train(replay_buffer)

            if done:
                break

        # Store the score for this episode
        scores.append(score)

        # Print the score every 100 episodes
        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-100:])))

    return scores
