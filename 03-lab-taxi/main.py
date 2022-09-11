from agent import Agent
from monitor import interact
import gym
import numpy as np
import time

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

# Watch game play simulation
env = gym.make('Taxi-v3', render_mode='human')
for i_episode in range(100):
    state = env.reset()

    while True:
    # agent selects an action
        action = agent.select_action(state)
        # agent performs the selected action
        next_state, reward, done, _ = env.step(action)
        # agent performs internal updates based on sampled experience
        agent.step(state, action, reward, next_state, done)
        
        env.render()
        time.sleep(0.001)
        
        # update the state (s <- s') to next time step
        state = next_state
        if done:
        # save final sampled reward
            break

env.close()
