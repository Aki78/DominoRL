from DominoEnv import DominoEnv
from train import DQNAgent
from train import train


env = DominoEnv()
agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64)
scores = train(env, agent, num_episodes=1000, max_steps=100)

