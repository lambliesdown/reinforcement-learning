# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:36:37 2020

@author: Karsten
"""
import numpy as np

def argmax(q_values):
    """
    Takes in a list of q_values and returns the index
    of the item with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value, then update top and reset ties to zero
        # if a value is equal to top value, then add the index to ties (hint: do this no matter what)
        # return a random selection from ties. (hint: look at np.random.choice)
        if q_values[i] == top:
            ties.append(i)
        elif q_values[i] > top:
            top = q_values[i]
            ties = [i]

    return np.random.choice(ties)
