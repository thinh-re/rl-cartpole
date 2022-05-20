# Introduction
Basic Reinforcement Learning: Deep Q Network on CartPole problem (OpenAI Gym) <br>

# Reinforcement learning
 - ε-policy
 - Bellman equation
 - Deep Q networks (2 hidden layers)
 - Replay buffer

# Environments
## Train
Use Google Colab (with GPU enabled) to train neural networks (`CartPole.ipynb`)

## Test
Ubuntu 22.04 LTS <br>
python 3.8.0 <bR>
pytorch 1.11.0 <br>

`python main.py`

Folder `pretrained_models` contains pretrained models with different epsilons (0.01, 0.05, 0.1, 0.5) after over 10000 iterations

# Results
![alt text](10000iters.png)

# References
The code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py