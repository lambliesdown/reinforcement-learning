# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:48:01 2020

@author: Karsten
"""

from .dyna_q_agent import DynaQAgent
import numpy as np

class DynaQPlusAgent(DynaQAgent):
    
    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction
                kappa (float): The scaling factor for the reward bonus

                seed (int): the seed for the RNG used in epsilon-greedy
                planning_seed (int): the seed for the RNG used in the planner
            }
        """

        # First, we get the relevant information from agent_info 
        # Note: we use np.random.RandomState(seed) to set the two different RNGs
        # for the planner and the rest of the code
        super().agent_init(agent_info)
        
        self.kappa = agent_info.get("kappa", 0.001)

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, tau, etc.
        # The visitation-counts can be stored as a table as well, like the action values 
        self.tau = np.zeros((self.num_states, self.num_actions))
        
    def update_model(self, past_state, past_action, state, reward):
        """updates the model 
    
        Args:
            past_state  (int): s
            past_action (int): a
            state       (int): s'
            reward      (int): r
        Returns:
            Nothing
        """
    
        # Recall that when adding a state-action to the model, if the agent is visiting the state
        #    for the first time, then the remaining actions need to be added to the model as well
        #    with zero reward and a transition into itself. Something like:
        ##   for action in self.actions:
        ##       if action != past_action:
        ##           self.model[past_state][action] = (past_state, 0)  
        #
        # Note: do *not* update the visitation-counts here. We will do that in `agent_step`.
        #
        # (3 lines)
    
        if past_state not in self.model:
            self.model[past_state] = {past_action : (state, reward)}
            for action in self.actions:
                if action != past_action:
                    self.model[past_state][action] = (past_state, 0)
        else:
            self.model[past_state][past_action] = (state, reward)
            
    def planning_step(self):
        """performs planning, i.e. indirect RL.
    
        Args:
            None
        Returns:
            Nothing
        """
        
        # The indirect RL step:
        # - Choose a state and action from the set of experiences that are stored in the model. (~2 lines)
        # - Query the model with this state-action pair for the predicted next state and reward.(~1 line)
        # - **Add the bonus to the reward** (~1 line)
        # - Update the action values with this simulated experience.                            (2~4 lines)
        # - Repeat for the required number of planning steps.
        #
        # Note that the update equation is different for terminal and non-terminal transitions. 
        # To differentiate between a terminal and a non-terminal next state, assume that the model stores
        # the terminal state as a dummy state like -1
        #
        # Important: remember you have a random number generator 'planning_rand_generator' as 
        #     a part of the class which you need to use as self.planning_rand_generator.choice()
        #     For the sake of reproducibility and grading, *do not* use anything else like 
        #     np.random.choice() for performing search control.
    
        for i in range(self.planning_steps):
            
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))
    
            s_new, r = self.model[s][a]
            if s_new != -1:
                target = r + self.kappa*np.sqrt(self.tau[s,a])+ self.discount*np.max(self.q_values[s_new])
            else:
                target = r + self.kappa*np.sqrt(self.tau[s,a])
    
            self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])


    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent is taking.
        """  
        
        # Update the last-visited counts (~2 lines)
        # - Direct-RL step (1~3 lines)
        # - Model Update step (~1 line)
        # - `planning_step` (~1 line)
        # - Action Selection step (~1 line)
        # Save the current state and action before returning the action to be performed. (~2 lines)
        
        # update the tau
        self.tau += 1
        s, a = self.past_state, self.past_action
        self.tau[s, a] = 0
        
        target = reward + self.discount*np.max(self.q_values[state])
        self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])
        
        self.update_model(s,a, state, reward)
        self.planning_step()
        
        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        
        return self.past_action
    
    def agent_end(self, reward):
        """Called when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Again, add the same components you added in agent_step to augment Dyna-Q into Dyna-Q+
        
        self.tau += 1
        s,a = self.past_state, self.past_action
        self.tau[s, a] = 0
        
        target = reward
        self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])
        
        self.update_model(s,a,-1,reward)
        self.planning_step()
