#!/usr/bin/env python

from rllib.utils import argmax
from rllib.rl_glue import RLGlue
from bandit_agent import BanditAgent
from ten_arm_env import TenArmEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class GreedyBanditAgent(BanditAgent):

    def agent_init(self, agent_info={}):
        super().agent_init(agent_info)
        num_arms = agent_info.get("num_actions", 1)
        self.arm_count = [0.0 for _ in range(num_arms)]
        
    def agent_step(self, reward, observation):
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        #######################
        
        # current action = ? # Use the argmax function you created above
        # (~2 lines)
        ### START CODE HERE ###
        current_action = argmax(self.q_values)
        ### END CODE HERE ###
        
        # Update action values. Hint: Look at the algorithm in section 2.4 of the textbook.
        # Increment the counter in self.arm_count for the action from the previous time step
        # Update the step size using self.arm_count
        # Update self.q_values for the action from the previous time step
        # (~3-5 lines)
        ### START CODE HERE ###
        self.arm_count[self.last_action] += 1
        step_size = 1/self.arm_count[self.last_action]
        self.q_values[self.last_action] += step_size*(reward - self.q_values[self.last_action])
        ### END CODE HERE ###
    
        self.last_action = current_action
        
        return current_action
        
def main():
    num_runs = 200                    # The number of times we run the experiment
    num_steps = 1000                  # The number of steps each experiment is run for
    env = TenArmEnvironment           # We the environment to use
    agent = GreedyBanditAgent         # We choose what agent we want to use
    agent_info = {"num_actions": 10}  # Pass the agent the information it needs; 
                                      # here it just needs the number of actions (number of arms).
    env_info = {}                     # Pass the environment the information it needs; in this case, it is nothing.
    
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