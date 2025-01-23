import torch
from torch import nn
import torch.nn.functional as F

''' 
torch.nn.functional contains a lot of convolution, pooling, activation functions etc
'''
'''
The input layer in the DQN will consist of 12 nodes, basically the 12 possible observations(state) by the bird (written in the repository)
The number of hidden layers and the number of nodes in the hidden layers will be based on trial&error
The output layer in the DQN will consist of 2 nodes, the 2 possible actions.

'''
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(DQN, self).__init__()

        # define the hidden layer. Single hidden layer used for now.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # define the output layer. 
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    sample_network = DQN(state_dim, action_dim)
   # state = torch.randn(1, state_dim)
    state = torch.randn(10, state_dim)
    output = sample_network(state)
    print(output)


''' 
In line32, calling the parameter "1" means that the bird took 1 action and passed through 1 state.
However in pytorch, we can calculate for a whole batch of states at once. 
Instead, I can call "10" (line33) (and that means that the bird took 10 actions one by one and hence passed through 10 different states) and pytorch will calculate me 10 separate sets of Q values. 

When line32 was run, debugging led to the conclusion that the random state generated using the randn function was : 
    tensor([[ 0.6203, -0.3261, -1.2880, -0.2955,  0.3233, -1.5467, -0.0082,  1.3097,
            -0.7764, -0.0593,  1.8870, -0.2358]])
& the shape of the state is ofcourse [1,12]. 
And the Q value generated as the output of the network was : tensor([[0.3719, 0.1546]

Instead, when line33 was run, debugging led to the conclusion that the random state generated using the randn function was : 
    tensor([[ 1.5128e+00,  1.5497e+00,  5.5617e-01,  1.6608e+00, -1.8403e+00,
         -9.3417e-01, -7.9345e-01,  7.4212e-02, -1.9947e-01,  8.1658e-02,
         -1.6049e-01,  1.3938e-01],
        [-8.9091e-01,  4.4844e-01,  4.4528e-01,  2.2059e+00,  2.1652e-01,
         -1.3258e-03,  2.9645e-01, -5.9841e-01,  5.2901e-01, -1.1864e+00,
          8.8645e-01, -9.2072e-01],
        [ 1.0722e+00, -1.8352e+00, -3.0136e-01, -1.6933e+00,  3.9150e-01,
          6.9231e-01, -1.7391e-01,  1.2277e+00, -1.6720e+00, -1.7684e+00,
         -7.6279e-01, -5.6435e-01],
        [-4.5064e-01,  1.1801e+00, -1.0932e+00,  5.7582e-01,  6.7073e-01,
          4.4708e-01, -1.1508e-01, -4.6568e-01, -2.3593e-01,  9.3162e-01,
          1.4346e+00, -3.5448e-01],
        [ 5.0252e-01,  4.5362e-01, -1.8089e+00, -1.2924e+00, -3.9225e-01,
         -1.3503e+00, -2.8584e+00, -5.4169e-03,  4.1654e-01,  4.3919e-01,
          5.2092e-01, -2.0178e+00],
        [ 5.8615e-01, -4.9824e-01, -1.5842e+00,  1.0366e+00,  8.1413e-01,
          1.6905e+00,  4.4145e-01,  1.7153e-01, -2.6279e+00, -3.9193e-01,
          7.0881e-01, -1.1257e-01],
        [-1.2246e+00, -6.8976e-01, -7.2023e-01,  9.8741e-01,  3.3892e-01,
         -1.6393e+00,  1.2972e+00, -2.4750e+00,  7.4067e-01,  7.7413e-01,
         -2.5217e-01,  7.5520e-01],
        [-2.9900e-01,  4.0180e-01,  9.5090e-01, -4.8504e-01,  1.6718e+00,
          5.2783e-01,  3.1490e-01, -8.3249e-01,  3.8185e-01, -4.4574e-01,
         -1.2206e+00, -1.1311e+00],
        [ 1.4055e-01, -5.6928e-01, -1.4841e+00, -4.4851e-01,  7.3421e-01,
          1.4556e+00,  6.0087e-01,  1.3035e+00,  1.1194e+00,  3.1033e-01,
          5.6472e-01, -1.1425e+00],
        [ 6.4357e-01,  1.1174e-01, -6.6354e-01, -3.9922e-01,  5.5349e-01,
         -1.2366e-01, -4.8976e-01, -2.3305e-01, -1.6557e+00, -1.2063e+00,
          3.8514e-01,  1.3229e+00]])
& the shape of the state is ofcourse [10,12]. 
And the Q value generated as the output of the network was :
 
    tensor([[-0.0040,  0.2574],
            [-0.0418,  0.2331],
            [-0.1622,  0.0220],
            [ 0.2843, -0.3365],
            [ 0.2304,  0.0650],
            [ 0.4165,  0.0447],
            [-0.0694,  0.3477],
            [ 0.3300, -0.0064],
            [ 0.1936, -0.4127],
            [ 0.3827, -0.1873]],
'''