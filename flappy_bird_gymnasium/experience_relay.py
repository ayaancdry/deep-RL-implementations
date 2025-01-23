'''
An "experience" (or transition) consists of the following : 
state, action (taken by the agent), new_state (after taking the action), reward (given after the action) & terminated (episode over or not).

The experiences are stored in a deque (double-ended queue). A deque keeps on getting filled until it reached its maximum length, after which it starts to remove the old elements and add the new ones.
'''

from collections import deque
import random

# Class to store experiences collected by an agent while interacting with the environment
class ReplayMemory():
    def __init__(self, maxlen, seed=None): 
        '''
        maxlen : maximum number of transitions the deque can store.
        seed : parameter to set a random seed to ensure reproducibility.
        reproducibility : "when you run the same algorithm on the same dataset with the same parameters, you should obtain the same or very similar results"
        '''
        self.memory = deque([], maxlen=maxlen)

        # optional seed for reproducibility. If seed is provided, then set a random seed. 
        if seed is not None:
            random.seed(seed)

    # Method to add a new transition to the replay-memory. 
    def append(self, transition):
        self.memory.append(transition)

    # Method to randomly sample a specified number of transitions from the replay-memory.
    # The randomly selected samples are imp for training RL agents.
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    # Method to get the number of transitions stored in the replay memory.
    def __len__(self):
        return len(self.memory)