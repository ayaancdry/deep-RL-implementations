import gymnasium as gym
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", render_mode = "human", use_lidar=True)
''' 
- render_mode='human' to render the environment continuously in the current terminal window
- use_lidar=False
'''

# initialise the environment using reset()
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

