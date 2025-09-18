import mani_skill.envs  # noqa: F401

from typing import Any, Tuple
import torch
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

from llm_tale.envs.maniskill_envs.utils.parse_affordance import Obj
from llm_tale.envs.maniskill_envs.utils.skrl_wrapper import Wrapper

from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig


class PickCubeEnv(Wrapper):
    def __init__(
        self,
        LLM_instuction=False,
        eval: bool = False,
        episode_length: int = 150,
        image_input=False,
        expert_reward=False,
        reward_scale=1,
        on_policy=True,
        record=False,
    ) -> None:
        reward_mode = "dense" if expert_reward else "sparse"
        obs_mode = "state_dict" if not image_input else "rgb"
        if record:
            pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
            human_render_camera_configs = CameraConfig("render_camera", pose, 1080, 1080, 1, 0.01, 100).__dict__
        else:
            human_render_camera_configs = None
        env = gym.make(
            "PickCube-v1",
            reward_mode=reward_mode,
            obs_mode=obs_mode,
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            render_mode="rgb_array",
            human_render_camera_configs=human_render_camera_configs,
        )
        super().__init__(
            env=env,
            LLM_instuction=LLM_instuction,
            eval=eval,
            episode_length=episode_length,
            image_input=image_input,
            expert_reward=expert_reward,
            reward_scale=reward_scale,
            on_policy=on_policy,
        )
        self.set_objects()

    def set_objects(self):
        self.task_objects = {
            "CubeA": Obj("CubeA", shape=[0.04, 0.04, 0.04]),
            "target": Obj("target", shape=None),
        }

    def define_observation_space(self):
        if self.image_input:
            state_num = 2 + 7 + 7 + 9
            if self.LLM_instuction:
                state_num += 7
            observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float),
                    "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                }
            )
        else:
            state_num = 2 + 7 + 7 + 10
            if self.LLM_instuction:
                state_num += 7 + 9
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float)
        return observation_space

    def get_object_state(self, obs_dict):
        if self.image_input:
            obj_pose = self._env.cube.pose.raw_pose[0]
        else:
            obj_pose = obs_dict["extra"]["obj_pose"][0]
        target_pose = obs_dict["extra"]["goal_pos"][0]
        self.objects = {"CubeA": obj_pose, "target": target_pose}
        for obj_name, obj in self.task_objects.items():
            obj.update_pose(self.objects[obj_name])
        # update object class
        obj_states = torch.cat([obj_pose, target_pose])
        return obj_states

    def get_picked_obj(self):
        return self._env.cube

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, reward, terminated, truncated, info

    def get_task_relevent_info(self, info):
        return info

    def task_relevant_reward(self, info):
        reward = 0
        terminated = False
        if info["success"]:
            terminated = True
            if not self.on_policy:
                info["progress_reward"] = 100
            else:
                info["progress_reward"] = 10 + 3 * (self.episode_length - self.step_num)
        else:
            info["progress_reward"] = 0
        reward += info["progress_reward"]
        return reward, torch.tensor(terminated)


env = PickCubeEnv
Instruction = "Pick the cubeA and move it to target"
Objects = """
Objects:
    CubeA = Object(position,orientation=None)
    target = Object(position,orientation=None)
"""
task = "PickCube"
llm_plan_path = f"cache/{task.lower()}_code_gpt-4o.pkl"
episode_length = 30
env_kwargs = {
    "ppo_explo": {
        "episode_length": episode_length,
        "LLM_instuction": True,
    },
    "td3_explo": {
        "episode_length": episode_length,
        "LLM_instuction": True,
        "on_policy": False,
        "reward_scale": 0.01,
    },
}
llm_tale_kwargs = {
    "ppo_explo": {
        "env_type": "maniskill",
    },
    "td3_explo": {
        "env_type": "maniskill",
    },
}
kwargs = {
    "ppo_explo": {
        "train": {
            "n_steps": 1024,
            "learning_rate": 3e-4,
            "start_std": -1,
            "device": "cuda",
            "tb_path": "exps_llm_tale/maniskill/{}/PPO".format(task),
        },
        "eval": {
            "start_std": -10,
            "device": "cuda",
            "tb_path": None,
        },
    },
    "td3_explo": {
        "train": {
            "exploration_noise": 0.2,
            "discount": 0.95,
            "gradient_steps": 1,
            "buffer_size": 500000,
            "learning_starts": 10000,
            "tb_path": "exps_llm_tale/maniskill/{}/TD3".format(task),
        },
        "eval": {
            "exploration_timesteps": -1,
            "learning_starts": 0,
            "tb_path": None,
        },
    },
}
