# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:40:11 2020

@author: Karsten
"""

from rllib.agent.q_learning_agent import QLearningAgent
from rllib.agent.expected_sarsa_agent import ExpectedSarsaAgent
from rllib.environment.cliff_walk_environment import CliffWalkEnvironment
from rllib.rl_glue import RLGlue
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem
import numpy as np
#import cliffworld_env

def main():

    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'figure.figsize': [10,5]})

    agents = {
        "Q-learning": QLearningAgent,
        "Expected Sarsa": ExpectedSarsaAgent
    }
    env = CliffWalkEnvironment #cliffworld_env.Environment
    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {} # Contains state visit counts during the last 10 episodes
    agent_info = {"num_actions": 4, "num_states": 48, "epsilon": 0.1, "step_size": 0.5, "discount": 1.0}
    env_info = {}
    num_runs = 100 # The number of runs
    num_episodes = 500 # The number of episodes in each run
    
    for algorithm in ["Q-learning", "Expected Sarsa"]:
        all_reward_sums[algorithm] = []
        all_state_visits[algorithm] = []
        for run in tqdm(range(num_runs)):
            agent_info["seed"] = run
            rl_glue = RLGlue(env, agents[algorithm])
            rl_glue.rl_init(agent_info, env_info)
    
            reward_sums = []
            state_visits = np.zeros(48)
            for episode in range(num_episodes):
                if episode < num_episodes - 10:
                    # Runs an episode
                    rl_glue.rl_episode(0) 
                else: 
                    # Runs an episode while keeping track of visited states
                    state, action = rl_glue.rl_start()
                    state_visits[state] += 1
                    is_terminal = False
                    while not is_terminal:
                        reward, state, action, is_terminal = rl_glue.rl_step()
                        state_visits[state] += 1
                    
                reward_sums.append(rl_glue.rl_return())
                
            all_reward_sums[algorithm].append(reward_sums)
            all_state_visits[algorithm].append(state_visits)
    

    for algorithm in ["Q-learning", "Expected Sarsa"]:
        plt.plot(np.mean(all_reward_sums[algorithm], axis=0), label=algorithm)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode",rotation=0, labelpad=40)
    plt.xlim(0,500)
    plt.ylim(-100,0)
    plt.legend()
    plt.show()

    for algorithm, position in [("Q-learning", 211), ("Expected Sarsa", 212)]:
        plt.subplot(position)
        average_state_visits = np.array(all_state_visits[algorithm]).mean(axis=0)
        # CliffWalkEnvironment is upside down vs. cliffworld_env
        grid_state_visits = np.flipud(average_state_visits.reshape((4,12)))
        grid_state_visits[0,1:-1] = np.nan
        plt.pcolormesh(grid_state_visits, edgecolors='gray', linewidth=2)
        plt.title(algorithm)
        plt.axis('off')
        cm = plt.get_cmap()
        cm.set_bad('gray')
    
        plt.subplots_adjust(bottom=0.0, right=0.7, top=1.0)
        cax = plt.axes([0.85, 0.0, 0.075, 1.])
        
    cbar = plt.colorbar(cax=cax)
    cbar.ax.set_ylabel("Visits during\n the last 10\n episodes", rotation=0, labelpad=70)
    plt.show()
       
    agents = {
        "Q-learning": QLearningAgent,
        "Expected Sarsa": ExpectedSarsaAgent
    }
    
    env = CliffWalkEnvironment
    all_reward_sums = {}
    step_sizes = np.linspace(0.1,1.0,10)
    agent_info = {"num_actions": 4, "num_states": 48, "epsilon": 0.1, "discount": 1.0}
    env_info = {}
    num_runs = 100
    num_episodes = 100
    all_reward_sums = {}
    
    for algorithm in ["Q-learning", "Expected Sarsa"]:
        for step_size in step_sizes:
            all_reward_sums[(algorithm, step_size)] = []
            agent_info["step_size"] = step_size
            for run in tqdm(range(num_runs)):
                agent_info["seed"] = run
                rl_glue = RLGlue(env, agents[algorithm])
                rl_glue.rl_init(agent_info, env_info)
    
                return_sum = 0
                for episode in range(num_episodes):
                    rl_glue.rl_episode(0)
                    return_sum += rl_glue.rl_return()
                all_reward_sums[(algorithm, step_size)].append(return_sum/num_episodes)
            
    
    for algorithm in ["Q-learning", "Expected Sarsa"]:
        algorithm_means = np.array([np.mean(all_reward_sums[(algorithm, step_size)]) for step_size in step_sizes])
        algorithm_stds = np.array([sem(all_reward_sums[(algorithm, step_size)]) for step_size in step_sizes])
        plt.plot(step_sizes, algorithm_means, marker='o', linestyle='solid', label=algorithm)
        plt.fill_between(step_sizes, algorithm_means + algorithm_stds, algorithm_means - algorithm_stds, alpha=0.2)
    
    plt.legend()
    plt.xlabel("Step-size")
    plt.ylabel("Sum of\n rewards\n per episode",rotation=0, labelpad=50)
    plt.xticks(step_sizes)
    plt.show()

   
if __name__ == "__main__":
    main()