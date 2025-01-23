import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN

class Agent:
    '''run function will do both the training as well as run the test afterwards'''
    def run(self, is_training=True, render=False):
        # env = gym.make("FlappyBird-v0", render_mode = "human" if render else None, use_lidar=True)
        ''' CartPole for easier & faster implementations'''
        env = gym.make("CartPole-v1", render_mode = "human" if render else None) 
        # render only if render variable is True, otherwise don't

        ''' 
        - render_mode='human' to render the environment continuously in the current terminal window
        - use_lidar=False
        '''

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        ''' 
        num_states = 12 and num_actions=2 in case of FlappyBird
        '''

        '''declare the policy dqn'''
        policy_dqn = DQN(num_states, num_actions)


        '''initialise the environment using reset()'''
        obs, _ = env.reset()


        while True:
            action = env.action_space.sample() # sampling a random action from the action space
            ''' 
            The sample space here is : 
            0 - do nothing 
            1 - flap
            '''
            obs, reward,terminated, _ , info = env.step(action) 
            ''' 
            the action will be passed to the step() function, which in return will give us  : 
            - observation made (what the next state is)
            - reward for the previous action
            - terminated will be returned TRUE if the bird hit the ground or any pipe
            - info : given for debugging
            '''

            if terminated:
                break

        env.close()

