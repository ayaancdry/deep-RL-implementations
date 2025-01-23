import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN
from experience_relay import ReplayMemory
import itertools

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
        rewards_per_episode = [] # to store the rewards of each episode in a list

        ''' 
        num_states = 12 and num_actions=2 in case of FlappyBird
        '''

        '''declare the policy dqn'''
        policy_dqn = DQN(num_states, num_actions)

        ''' If we're training, initialise the deque, i.e, the replay memory'''
        if is_training:
            memory = ReplayMemory(10000)
        

        for episode in itertools.count():
            # itertools help in running the loop continuously till the time I stop it after I get decent results. 

            ''' --- A single Episode begins here ---'''

            '''initialise the environment using reset()'''
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0 # Initialise reward in each episode to zero


            while not terminated:
                action = env.action_space.sample() # sampling a random action from the action space
                ''' 
                The sample space here is : 
                0 - do nothing 
                1 - flap
                '''
                new_state, reward, terminated, _ , info = env.step(action) 
                ''' 
                the action will be passed to the step() function, which in return will give us  : 
                - observation made (what the next state is)
                - reward for the previous action
                - terminated will be returned TRUE if the bird hit the ground or any pipe
                - info : given for debugging
                '''

                episode_reward += reward

                ''' If we'e training, then append the following to the memory '''
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                # Move to new state. Keep track of your current state
                state = new_state


                # if terminated:
                #     break
            
            rewards_per_episode.append(episode_reward) # Append the reward of each episode to the list

            ''' --- A Single episode ends here --- '''


