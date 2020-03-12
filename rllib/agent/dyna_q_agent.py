# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:23:06 2020

@author: Karsten
"""

from .base_agent import BaseAgent
import numpy as np

class DynaQAgent(BaseAgent):

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

                seed (int): the seed for the RNG used in epsilon-greedy
                planning_seed (int): the seed for the RNG used in the planner
            }
        """
        super().agent_init(agent_info)
        
        # First, we get the relevant information from agent_info 
        # NOTE: we use np.random.RandomState(seed) to set the two different RNGs
        # for the planner and the rest of the code
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.planning_steps = agent_info.get("planning_steps", 10)

        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        # A simple way to implement the model is to have a dictionary of dictionaries, 
        #        mapping each state to a dictionary which maps actions to (reward, next state) tuples.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {} # model is a dictionary of dictionaries, which maps states to actions to 
                        # (reward, next_state) tuples
                        
    def update_model(self, past_state, past_action, state, reward):
        """updates the model 
        
        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """
        # Update the model with the (s,a,s',r) tuple (1~4 lines)
        if past_state not in self.model:
            self.model[past_state] = { past_action : (state,reward) }
        else:
            self.model[past_state][past_action] = (state,reward)
            
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
                target = r + self.discount*np.max(self.q_values[s_new])
            else:
                target = r
    
            self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
    
        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()
    
        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """
    
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)
    
        return action
    
    def agent_start(self, state):
        """The first method called when the experiment starts, 
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        
        self.past_action = self.choose_action_egreedy(state)
        self.past_state = state   
        
        return self.past_action
    
    def agent_step(self, reward, state):
        """A step taken by the agent.
    
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        
        # - Direct-RL step (~1-3 lines)
        # - Model Update step (~1 line)
        # - `planning_step` (~1 line)
        # - Action Selection step (~1 line)
        # Save the current state and action before returning the action to be performed. (~2 lines)
    
        s, a = self.past_state, self.past_action
        target = reward + self.discount*np.max(self.q_values[state])
        self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])
        
        self.update_model(s, a, state, reward)
        self.planning_step()
        
        self.past_action = self.choose_action_egreedy(state)
        self.past_state = state
        
        return self.past_action
    
    def agent_end(self, reward):
        """Called when the agent terminates.
    
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        
        # - Direct RL update with this final transition (1~2 lines)
        # - Model Update step with this final transition (~1 line)
        # - One final `planning_step` (~1 line)
        #
        # Note: the final transition needs to be handled carefully. Since there is no next state, 
        #       you will have to pass a dummy state (like -1), which you will be using in the planning_step() to 
        #       differentiate between updates with usual terminal and non-terminal transitions.
    
        s, a = self.past_state, self.past_action
        target = reward
        self.q_values[s, a] += self.step_size*(target - self.q_values[s, a])
        
        self.update_model(s, a, -1, reward)
        self.planning_step()
