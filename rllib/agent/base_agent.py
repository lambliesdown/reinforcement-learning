#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from .agent import Agent
import numpy as np

class BaseAgent(Agent):

    def agent_init(self, agent_info={}):
        
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount", 0.95)
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size", 0.1)
        # The parameter for epsilon-greedy exploration,
        self.epsilon = agent_info.get("epsilon", 0.1)
        # Seed for random number generator
        self.seed = agent_info.get("seed", None)
        
        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        self.rand_generator = np.random.RandomState(self.seed)

    def argmax(self, q_values):
        """argmax with random tie-breaking. Use the class specific version with this class' random generator
        for reproducible behavior.
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
