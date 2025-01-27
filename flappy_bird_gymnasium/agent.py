import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN
from experience_relay import ReplayMemory
import itertools
import yaml
import random

class Agent:
    
    ''' open the .yaml file and read the hyperparameters and store them into an array'''
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]  # store the hyperparameters read from the file into an array

            self.replay_memory_size = hyperparameters['replay_memory_size'] # size of replay memory
            self.mini_batch_size = hyperparameters['mini_batch_size']      # size of training dataset sampled from the replay memory
            self.epsilon_init = hyperparameters['epsilon_init']             # 1-100% random actions
            self.epsilon_decay = hyperparameters['epsilon_decay']           # epsilon decay rate
            self.epsilon_min = hyperparameters['epsilon_min']               # minimum epsilon value
    

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
        epsilon_history = [] # to store the values taken by epsilon throughout all episodes

        ''' 
        num_states = 12 and num_actions=2 in case of FlappyBird
        '''

        '''declare the policy dqn'''
        policy_dqn = DQN(num_states, num_actions)

        ''' If we're training, initialise the deque, i.e, the replay memory'''
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
        
            epsilon = self.epsilon_init # initialised epsilon


        for episode in itertools.count():
            # itertools help in running the loop continuously till the time I stop it after I get decent results. 

            ''' --- A single Episode begins here ---'''

            '''initialise the environment using reset()'''
            state, _ = env.reset()

            # convert into tensor for usage into pytorch
            state = torch.tensor(state, dtype=torch.float)

            terminated = False
            episode_reward = 0.0 # Initialise reward in each episode to zero


            while not terminated:
                ''' if we're training and we select a random number which is less than epsilon, then we a random action will be implemented'''
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample() 
                    # convert into tensor
                    action = torch.tensor(action, dtype=torch.int64)
            
                else: 
                    '''Over time, agent will keep on learning and then take the actions dictated by the trained policy.
                    .argmax() because in the dqn, the output layer has two neurons - each stating the possible actions (do nothing or flap). The output of each node will be a value in between 0 & 1 (probabilities), sum of the outputs will be 1.
                    We will choose that neuron ( 0 or 1 ; 0 : do nothing, 1 : flap ) which is higher, i.e, higher chances of taking that action. Hence .argmax() used
                    '''
                    with torch.no_grad():
                        ''' pytorch automatically does gradient calculations while training. Here, we're just evaluating, hence we can turn off the gradient calculations.'''
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                ''' 
                The sample space here is : 
                0 - do nothing 
                1 - flap
                '''
                new_state, reward, terminated, _ , info = env.step(action.item()) 
                ''' 
                the action will be passed to the step() function, which in return will give us  : 
                - observation made (what the next state is)
                - reward for the previous action
                - terminated will be returned TRUE if the bird hit the ground or any pipe
                - info : given for debugging
                '''

                episode_reward += reward

                # convert into tensor
                new_state = torch.tensor(new_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                ''' If we'e training, then append the following to the memory '''
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                # Move to new state. Keep track of your current state
                state = new_state
            
            rewards_per_episode.append(episode_reward) # Append the reward of each episode to the list

            ''' --- A Single episode ends here --- '''

            # We need to slowly decease epsilon after one episode. 
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)


if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_training=True,render=True )

