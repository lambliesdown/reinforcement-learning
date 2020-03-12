#!/usr/bin/env python

from .base_agent import BaseAgent

import numpy as np

class BanditAgent(BaseAgent):
  """agent does *no* learning, randomly choose an action"""

  def agent_init(self, agent_info={}):
    super().agent_init(agent_info)

    self.initial_value = agent_info.get("initial_value", 0.0)
    self.q_values = np.ones(self.num_actions) * self.initial_value

    self.last_action = 0
    self.arm_count = [0.0 for _ in range(self.num_actions)]

  def agent_start(self, observation):
    self.last_action = np.random.choice(self.num_actions)  
    return self.last_action

  def agent_step(self, reward, observation):
    self.last_action = np.random.choice(self.num_actions)
    return self.last_action

  def agent_end(self, reward):
    pass

  def agent_cleanup(self):
    pass

  def agent_message(self, message):
    pass


class GreedyBanditAgent(BanditAgent):

  def agent_step(self, reward, observation):
    current_action = self.get_current_action()
    
    # Update action values. Hint: Look at the algorithm in section 2.4 of the textbook.
    # Increment the counter in self.arm_count for the action from the previous time step
    # Update the step size using self.arm_count
    # Update self.q_values for the action from the previous time step
    self.arm_count[self.last_action] += 1
    step_size = 1/self.arm_count[self.last_action]
    self.q_values[self.last_action] += step_size*(reward - self.q_values[self.last_action])

    self.last_action = current_action
    
    return current_action

  def get_current_action(self):
    return self.argmax(self.q_values)    



class EpsilonGreedyBanditAgent(GreedyBanditAgent):
    
  def get_current_action(self):
    if np.random.random() < self.epsilon:
      return np.random.choice(range(len(self.arm_count)) )
    else:
      return self.argmax(self.q_values)