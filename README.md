# Deep Q Network on CartPole problem (OpenAI Gym)

## Objectives

- Understand the trade-off between exploitation and exploration with ε-policy

## Reinforcement learning

- ε-policy
- Bellman equation
- Deep Q networks (2 hidden layers)
- Replay buffer

## Environments

### Train

Use Google Colab (with GPU enabled) to train neural networks (`CartPole.ipynb`)

4 trained models (after over 10000 iterations) are located in the directory `pretrained_models`:

- `0.1-ckpt.pth`: ε = 0.1
- `0.01-ckpt.pth`: ε = 0.01
- `0.5-ckpt.pth`: ε = 0.5
- `0.05-ckpt.pth`: ε = 0.05

### Test

- Ubuntu 22.04 LTS
- Python 3.8.0
- Pytorch 1.11.0

Command:

```bash
python main.py
```

## Results

Average reward (moving average with window=500) for different epsilons after 10000 iterations
![alt text](10000iters.png)

- Greedy test: Model trained with ε=0.01
![alt text](results/0.01-result.gif)

- Greedy test: Model trained with ε=0.05
![alt text](results/0.05-result.gif)

- Greedy test: Model trained with ε=0.1
![alt text](results/0.1-result.gif)

- Greedy test: Model trained with ε=0.5
![alt text](results/0.5-result.gif)

## References

The code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py