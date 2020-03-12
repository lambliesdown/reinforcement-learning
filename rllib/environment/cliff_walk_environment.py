# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:42:47 2020

@author: Karsten
"""

from .environment import Environment

from operator import add

class CliffWalkEnvironment(Environment):
    
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        
        # Note, we can setup the following variables later, in env_start() as it is equivalent. 
        # Code is left here to adhere to the note above, but these variables are initialized once more
        # in env_start() [See the env_start() function below.]
        
        reward = None
        state = None # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)
        
        # AN ASIDE: Observation is a general term used in the RL-Glue files that can be interachangeably 
        # used with the term "state" for our purposes and for this assignment in particular. 
        # A difference arises in the use of the terms when we have what is called Partial Observability where 
        # the environment may return states that may not fully represent all the information needed to 
        # predict values or make decisions (i.e., the environment is non-Markovian.)
        
        # Set the default height to 4 and width to 12 (as in the diagram given above)
        self.grid_h = env_info.get("grid_height", 4) 
        self.grid_w = env_info.get("grid_width", 12)
        
        # Now, we can define a frame of reference. Let positive x be towards the direction down and 
        # positive y be towards the direction right (following the row-major NumPy convention.)
        # Then, keeping with the usual convention that arrays are 0-indexed, max x is then grid_h - 1 
        # and max y is then grid_w - 1. So, we have:
        # Starting location of agent is the bottom-left corner, (max x, min y). 
        self.start_loc = (self.grid_h - 1, 0)
        # Goal location is the bottom-right corner. (max x, max y).
        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)
        
        # The cliff will contain all the cells between the start_loc and goal_loc.
        self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.
    
        Returns:
            The first state from the environment.
        """
        reward = 0
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)
    
        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.
    
        Args:
            action: The action taken by the agent
    
        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """
    
        if action == 0: # UP (Task 1)
            possible_next_loc = tuple(map(add, self.agent_loc, (-1, 0)))
            if possible_next_loc[0] >= 0 :
                self.agent_loc = possible_next_loc
            else:
                pass 
        elif action == 1: # LEFT
            possible_next_loc = tuple(map(add, self.agent_loc, (0, -1)))
            if possible_next_loc[1] >= 0: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 2: # DOWN
            possible_next_loc = tuple(map(add, self.agent_loc, (+1, 0)))
            if possible_next_loc[0] < self.grid_h: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        elif action == 3: # RIGHT
            possible_next_loc = tuple(map(add, self.agent_loc, (0, +1)))
            if possible_next_loc[1] < self.grid_w: # Within Bounds?
                self.agent_loc = possible_next_loc
            else:
                pass # Stay.
        else: 
            raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")
    
        reward = -1
        terminal = False
    
        # Hint: Consider the initialization of reward and terminal variables above. Then, note the 
        # conditional statements and comments given below and carefully ensure to set the variables reward 
        # and terminal correctly for each case.
        if self.agent_loc == self.goal_loc: # Reached Goal!
            terminal = True
        
        elif self.agent_loc in self.cliff:# Fell into the cliff!
            self.agent_loc = self.start_loc
            reward = -100
            terminal = False
            
        else: 
            pass
        
        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.agent_loc = self.start_loc
        
    # helper method
    def state(self, loc):
        index = self.grid_w*loc[0] + loc[1]
        return index
