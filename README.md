# Code for Adversarial Diffusion for Robust Reinforcement Learning

## How to run the code

- We suggest creating a python 3.10 venv
- Once you have created your virtual environment, you activate it
- To train the agent, run the following commands from the folder `adversarial-world-models-main`

```
pip install -r ad-rrl_requirements.txt
pip install -e .
cd scripts
python online_rl.py --config config.online_rl.cheetah --seed 1 --run_number 1
```

## Selecting a different Mujoco training environment

To use another environment, change the last part of the config string (the one saying "cheetah") in the example above. In the `config/online_rl` folder you can find the IDs for the other environments (e.g., to train on hopper you will use `config.online_rl.cheetah`).