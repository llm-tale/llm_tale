"""Code based on SKRL examples: https://skrl.readthedocs.io/en/latest/intro/examples.html"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler

from llm_tale.agents.state_process import CpuRandomMemory
from llm_tale.agents.cnn import CnnEncoder


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.leaky_relu(self.linear_layer_1(inputs["states"]))
        x = F.leaky_relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), {}


class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


class ImageActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.num_states = self.observation_space.spaces["state"].shape[0]
        prop_dim = self.num_states
        proj_dim = 128
        self.cnn = CnnEncoder(obs_shape=[3, 64, 64], proj_dim=proj_dim)
        proj_dim = proj_dim + prop_dim

        self.net = nn.Sequential(nn.Linear(proj_dim, 400), nn.LeakyReLU(), nn.Linear(400, 300), nn.LeakyReLU())

        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        _input = inputs["states"]
        state = self.cnn.cnn_forward(_input)
        state = self.net(state)
        return torch.tanh(self.action_layer(state)), {}


class ImageCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        print("initializing image input network")
        self.num_states = self.observation_space.spaces["state"].shape[0]
        prop_dim = self.num_states + self.num_actions
        proj_dim = 128
        self.cnn = CnnEncoder(obs_shape=[3, 64, 64], proj_dim=proj_dim)
        proj_dim = proj_dim + prop_dim

        self.net = nn.Sequential(
            nn.Linear(proj_dim, 400), nn.LeakyReLU(), nn.Linear(400, 300), nn.LeakyReLU(), nn.Linear(300, 1)
        )

    def compute(self, inputs, role):
        _input = inputs["states"]
        action = inputs["taken_actions"]
        state = self.cnn.cnn_forward(_input, action)
        state = self.net(state)
        return state, {}


def setup_td3(
    env,
    batch_size=64,
    buffer_size=200000,
    discount=0.99,
    gradient_steps=1,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    exploration_noise=0.1,
    learning_starts=1000,
    grad_norm_clip=0.25,
    smooth_regularization_noise=None,
    state_preprocessor=False,
    device="cuda",
    image_input=False,
    tb_path=None,
    exp_name=None,
    wandb=False,
    checkpoint_interval=5000,
    exploration_timesteps=None,
):
    models = {}
    if not image_input:
        _ACTOR = DeterministicActor
        _CRITIC = DeterministicCritic
    else:
        _ACTOR = ImageActor
        _CRITIC = ImageCritic
    models["policy"] = _ACTOR(env.observation_space, env.action_space, device)
    models["target_policy"] = _ACTOR(env.observation_space, env.action_space, device)
    models["critic_1"] = _CRITIC(env.observation_space, env.action_space, device)
    models["critic_2"] = _CRITIC(env.observation_space, env.action_space, device)
    models["target_critic_1"] = _CRITIC(env.observation_space, env.action_space, device)
    models["target_critic_2"] = _CRITIC(env.observation_space, env.action_space, device)

    cfg_agent = TD3_DEFAULT_CONFIG.copy()
    cfg_agent["gradient_steps"] = gradient_steps
    cfg_agent["batch_size"] = batch_size
    cfg_agent["discount_factor"] = discount
    cfg_agent["exploration"]["noise"] = GaussianNoise(0, exploration_noise, device=device)
    cfg_agent["exploration"]["timesteps"] = exploration_timesteps
    cfg_agent["actor_learning_rate"] = actor_learning_rate
    cfg_agent["critic_learning_rate"] = critic_learning_rate
    cfg_agent["random_timesteps"] = learning_starts
    cfg_agent["learning_starts"] = learning_starts
    cfg_agent["grad_norm_clip"] = grad_norm_clip
    cfg_agent["smooth_regularization_noise"] = (
        GaussianNoise(0, smooth_regularization_noise, device=device) if smooth_regularization_noise else None
    )
    cfg_agent["state_preprocessor"] = RunningStandardScaler if state_preprocessor else None
    cfg_agent["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

    if tb_path:
        cfg_agent["experiment"]["write_interval"] = 500
        cfg_agent["experiment"]["checkpoint_interval"] = checkpoint_interval
        cfg_agent["experiment"]["directory"] = tb_path
    else:
        # we don't want to log anything
        cfg_agent["experiment"]["write_interval"] = 0
        cfg_agent["experiment"]["checkpoint_interval"] = 0
        cfg_agent["experiment"]["directory"] = None

    if wandb:
        cfg_agent["experiment"]["wandb"] = True
        cfg_agent["experiment"]["wandb_kwargs"] = {
            "project": "ExploRLLM",
            "reinit": True,
            "name": exp_name if exp_name is not None else "ExploRLLM",
        }
    cfg_agent["experiment"]["experiment_name"] = exp_name if exp_name is not None else ""

    _memory = CpuRandomMemory if image_input else RandomMemory
    memory = _memory(
        memory_size=buffer_size, num_envs=env.num_envs, device=device if not image_input else "cpu", replacement=True
    )

    agent = TD3(
        models=models,
        memory=memory,
        cfg=cfg_agent,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return agent
