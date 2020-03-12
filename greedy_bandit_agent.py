#!/usr/bin/env python

from rllib.rl_glue import RLGlue
from rllib.environment.bandit_environment import BanditEnvironment
from rllib.agent.bandit_agent import GreedyBanditAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def main():
    num_runs = 200                          # The number of times we run the experiment
    num_steps = 1000                        # The number of steps each experiment is run for
    num_arms = 10
    env = BanditEnvironment                 # We the environment to use
    agent = GreedyBanditAgent               # We choose what agent we want to use
    agent_info = {"num_actions": num_arms}  # Pass the agent the information it needs; 
                                            # here it just needs the number of actions (number of arms).
    env_info = {"num_arms": num_arms}    # Pass the environment the information it needs
    
    all_averages = []
    
    for i in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below once the code is run
        rl_glue = RLGlue(env, agent)          # Creates a new RLGlue experiment with the env and agent we chose above
        rl_glue.rl_init(agent_info, env_info) # Pass RLGlue what it needs to initialize the agent and environment
        rl_glue.rl_start()                    # Start the experiment
    
        scores = [0]
        averages = []
        
        for i in range(num_steps):
            reward, _, action, _ = rl_glue.rl_step() # The environment and agent take a step and return
                                                     # the reward, and action taken.
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
        all_averages.append(averages)
    
    plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
    plt.plot(np.mean(all_averages, axis=0))
    plt.legend(["Best Possible", "Greedy"])
    plt.title("Average Reward of Greedy Agent")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()

if __name__ == "__main__":
    main()