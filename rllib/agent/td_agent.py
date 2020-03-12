# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:33:43 2020

@author: Karsten
"""

from .base_agent import BaseAgent
import numpy as np

class TDAgent(BaseAgent):
    
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        super().agent_init(agent_info)
        
        # Policy will be given, recall that the goal is to accurately estimate its corresponding value function. 
        self.policy = agent_info.get("policy")
    
        # Initialize an array of zeros that will hold the values.
        # Recall that the policy can be represented as a (# States, # Actions) array. With the 
        # assumption that this is the case, we can use the first dimension of the policy to
        # initialize the array for values.
        self.values = np.zeros((self.policy.shape[0],))
        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        # The policy can be represented as a (# States, # Actions) array. So, we can use 
        # the second dimension here when choosing an action.
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step after the last action, i.e., where the agent ended up after the
                last action
        Returns:
            The action the agent is taking.
        """
        # Hint: We should perform an update with the last state given that we now have the reward and
        # next state. We break this into two steps. Recall for example that the Monte-Carlo update 
        # had the form: V[S_t] = V[S_t] + alpha * (target - V[S_t]), where the target was the return, G_t.
        target = reward + self.discount*self.values[state]
        self.values[self.last_state] = self.values[self.last_state] + self.step_size*(target - self.values[self.last_state])
         
        # Having updated the value for the last state, we now act based on the current 
        # state, and set the last state to be current one as we will next be making an 
        # update with it when agent_step is called next once the action we return from this function 
        # is executed in the environment.
    
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
    
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
    
        # Hint: Here too, we should perform an update with the last state given that we now have the 
        # reward. Note that in this case, the action led to termination. Once more, we break this into 
        # two steps, computing the target and the update itself that uses the target and the 
        # current value estimate for the state whose value we are updating.
        target = reward
        self.values[self.last_state] = self.values[self.last_state] + self.step_size*(target - self.values[self.last_state])

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.last_state = None
        
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_values":
            return self.values
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")