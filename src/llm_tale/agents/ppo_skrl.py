"""Code based on SKRL examples: https://skrl.readthedocs.io/en/latest/intro/examples.html"""

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from llm_tale.agents.state_process import ImageRunningStandardScaler
from llm_tale.agents.cnn import CnnEncoder
import itertools
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ExponentialLR


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        start_std=-0.5,
        min_log_std=-20,
        max_log_std=0,
        reduction="sum",
    ):
        print(observation_space)
        print(action_space)
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU()
        )

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(start_std * torch.ones(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        _input = inputs["states"]
        if role == "policy":
            self._shared_output = self.net(_input)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(_input) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


class SharedImageinput(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        start_std=-0.5,
        min_log_std=-20,
        max_log_std=0,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        self.num_states = self.observation_space.spaces["state"].shape[0]
        prop_dim = self.num_states
        proj_dim = 128
        self.cnn = CnnEncoder(obs_shape=[3, 64, 64], proj_dim=proj_dim)
        proj_dim = proj_dim + prop_dim

        self.net = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
        )

        self.mean_layer = nn.Linear(128, self.num_actions)

        self.log_std_parameter = nn.Parameter(start_std * torch.ones(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        _input = inputs["states"]
        if role == "policy":
            state = self.cnn.cnn_forward(_input)
            self._shared_output = self.net(state)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(self.cnn.cnn_forward(_input)) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


class PPO_reg(PPO):
    def __init__(self, models, memory, observation_space, action_space, device, cfg, l2_reg=0.00):
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.l2_reg = l2_reg
        if self.l2_reg > 0:
            self.setup_l2optimizer()

    def setup_l2optimizer(self):

        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate, weight_decay=self.l2_reg)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()),
                    lr=self._learning_rate,
                    weight_decay=self.l2_reg,
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer


def setup_ppo(
    env,
    n_steps=2048,
    discount=0.99,
    image_input=False,
    device="cuda",
    start_std=-0.5,
    learning_rate=1e-4,
    learning_rate_scheduler=None,
    learning_epochs=8,
    mini_batches=4,
    gae_lambda=0.95,
    grad_norm_clip=0.5,
    value_clip=0.4,
    ratio_clip=0.2,
    l2_reg=0.00,
    value_loss_scale=2.0,
    tb_path=None,
    checkpoint_interval=5000,
    exp_name=None,
    wandb=True,
):
    print("n_steps", n_steps)
    print("discount", discount)
    print("device", device)
    print("start_std", start_std)
    print("learning_rate", learning_rate)
    print("learning_rate_scheduler", learning_rate_scheduler)
    print("l2_reg", l2_reg)
    print("tb_path", tb_path)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = n_steps  # memory_size
    cfg["learning_epochs"] = learning_epochs
    cfg["mini_batches"] = mini_batches
    cfg["discount_factor"] = discount
    cfg["lambda"] = gae_lambda
    cfg["learning_rate"] = learning_rate
    if learning_rate_scheduler == "exp":
        cfg["learning_rate_scheduler"] = ExponentialLR
        cfg["learning_rate_scheduler_kwargs"] = {"gamma": 0.999}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = grad_norm_clip
    cfg["ratio_clip"] = ratio_clip
    cfg["value_clip"] = value_clip
    cfg["clip_predicted_values"] = True
    cfg["value_loss_scale"] = value_loss_scale
    cfg["kl_threshold"] = 0
    if not image_input:
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    else:
        cfg["state_preprocessor"] = ImageRunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    if tb_path:
        cfg["experiment"]["write_interval"] = 500
        cfg["experiment"]["checkpoint_interval"] = checkpoint_interval
        cfg["experiment"]["directory"] = tb_path
        if wandb:
            cfg["experiment"]["wandb"] = True
            cfg["experiment"]["wandb_kwargs"] = {
                "project": "ExploRLLM",
                "reinit": True,
                "name": exp_name if exp_name is not None else "ExploRLLM",
            }
        if exp_name is not None:
            cfg["experiment"]["experiment_name"] = exp_name
    else:
        # we don't want to log anything
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0
        cfg["experiment"]["wandb"] = False
        cfg["experiment"]["directory"] = None
    if image_input:
        memory = RandomMemory(memory_size=n_steps, num_envs=1, device=device)
    else:
        memory = RandomMemory(memory_size=n_steps, num_envs=1, device=device)
    models = {}
    if not image_input:
        models["policy"] = Shared(env.observation_space, env.action_space, device, start_std=start_std)
    else:
        models["policy"] = SharedImageinput(env.observation_space, env.action_space, device, start_std=start_std)
    models["value"] = models["policy"]
    if l2_reg > 0:
        agent = PPO_reg(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            l2_reg=l2_reg,
        )
    else:

        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    return agent
