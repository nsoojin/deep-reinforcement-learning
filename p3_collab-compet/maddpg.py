from ddpg_agent import ReplayBuffer
from ddpg_agent import Agent
import torch

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class MADDPG():
    def __init__(self, action_size, state_size, random_seed, discount_factor=0.95, tau=0.02):
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(2)]
        self.discount_factor = discount_factor
        self.tau = tau
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done, self.memory)

    def act(self, states, add_noise=True):
        return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, gamma):
        for agent, experience in zip(self.agents, experiences):
            agent.learn(experience, gamma)

    def save(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(i))
    
    def load(self):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor_{}.pth'.format(i)))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic_{}.pth'.format(i)))



    