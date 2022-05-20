from typing import Union, Tuple, List
from torch import Tensor

# imports
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH: str = 'pretrained_models/0.1-ckpt.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')

class QNetwork(nn.Module):
    def __init__(self, insize: int, outsize: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(insize, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, outsize)

        self.he_initialization(self.fc1)
        self.he_initialization(self.fc2)
        self.he_initialization(self.fc3)
        

    def he_initialization(self, layer: nn.Linear)->None:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

    def forward(self, x: Tensor)->Tensor:
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs: Tensor, epsilon: float)->int:
        '''
        Discrete action 1 or 0
        '''
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

env = gym.make('CartPole-v1')
s: np.ndarray = env.reset()

seed: int = 742
torch.manual_seed(seed)
# env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

q = QNetwork(np.array(env.observation_space.shape).prod(), env.action_space.n)
q.load_state_dict(torch.load(MODEL_PATH, map_location=device))
q_target = QNetwork(np.array(env.observation_space.shape).prod(), env.action_space.n)
q.to(device)
q_target.to(device)
q_target.load_state_dict(q.state_dict())

score: float = 0.0

env = gym.make('CartPole-v0')
s: np.ndarray = env.reset()
score = 0.0

env.render()

while True:
    obs: Tensor = torch.from_numpy(s).float().unsqueeze(0) # (1, 4)
    obs = obs.to(device)
    a = q.sample_action(obs, epsilon=0.0)
    s, r, done, _ = env.step(a)

    score += r

    env.render()

    if done:
        break

print('total score', score)
env.close()