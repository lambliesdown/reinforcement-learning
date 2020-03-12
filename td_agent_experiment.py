# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:40:11 2020

@author: Karsten
"""

from rllib.agent.td_agent import TDAgent
from rllib.environment.cliff_walk_environment import CliffWalkEnvironment
from rllib.rl_glue import RLGlue

import numpy as np
from manager import Manager

def run_experiment(env_info, agent_info, 
                   num_episodes=5000,
                   experiment_name=None,
                   plot_freq=100,
                   true_values_file=None,
                   value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)
    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0) # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    if true_values_file is not None:
        # Grading: The Manager will check that the values computed using your TD agent match 
        # the true values (within some small allowance) across the states. In addition, it also
        # checks whether the root mean squared value error is close to 0.
        manager.run_tests(values, value_error_threshold)
    
    return values

def main():
    env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
    agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}
    
    # The Optimal Policy that strides just along the cliff
    policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
    policy[36] = [1, 0, 0, 0]
    for i in range(24, 35):
        policy[i] = [0, 0, 0, 1]
    policy[35] = [0, 0, 1, 0]
    
    agent_info.update({"policy": policy})
    
    true_values_file = "optimal_policy_value_fn.npy"
    _ = run_experiment(env_info, agent_info, num_episodes=5000, experiment_name="Policy Evaluation on Optimal Policy",
                       plot_freq=500, true_values_file=true_values_file)
        

if __name__ == "__main__":
    main()