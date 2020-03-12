# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:12:34 2020

@author: Karsten
"""

from .base_agent import BaseAgent
import numpy as np

class QLearningAgent(BaseAgent):
    
    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.
        """
        
        super().agent_init(agent_info)
        
        # Store the parameters provided in agent_info.
        self.num_actions = agent_info["num_actions"]
        self.num_states = agent_info["num_states"]
        
        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Perform an update (1 line)
        target = reward + self.discount*np.max(self.q[state, :])
        self.q[self.prev_state, self.prev_action] += self.step_size*(target - self.q[self.prev_state,self.prev_action])
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode (1 line)
        target = reward
        self.q[self.prev_state, self.prev_action] += self.step_size*(target - self.q[self.prev_state,self.prev_action])
