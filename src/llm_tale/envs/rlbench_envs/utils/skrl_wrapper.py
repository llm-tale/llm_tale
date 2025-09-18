from typing import Any, Tuple

import torch
import numpy as np
from skrl.envs.wrappers.torch.base import Wrapper
from llm_tale.utils.rotation_utils import axisangle2quat
from llm_tale.envs.rlbench_envs.utils.robot import check_object_grasp
from llm_tale.llm_tale import LLM_TALE

from copy import deepcopy


class _Wrapper(Wrapper):
    def __init__(
        self,
        env: Any,
        LLM_instuction=False,
        eval: bool = False,
        episode_length: int = 150,
        image_input=False,
        expert_reward=False,
        reward_scale=1,
        on_policy=True,
        pos_only=False,
        render_mode="rgb_array",
    ) -> None:
        super().__init__(env)
        self.device = torch.device("cuda")
        self._vectorized = False
        self._obs_dict = None
        self.last_obs_dict = None
        self.eval = eval
        self.LLM_instuction = LLM_instuction
        self.image_input = image_input
        self.episode_length = episode_length
        self.expert_reward = expert_reward
        self.reward_scale = reward_scale
        self.on_policy = on_policy
        self.current_episode = 0
        self.pick_success_memory = torch.zeros(1000000, dtype=torch.bool)
        self.success_memory = torch.zeros(1000000, dtype=torch.float32)
        self.rl_policy = None
        self._observation_space = self.define_observation_space()

        self.pos_only = pos_only
        self.task_objects = None
        self.llm_tale = None
        self.render_mode = render_mode
        self._closed = False
        self.metadata = {}

    def set_llm_tale(
        self,
        llm_tale: LLM_TALE = None,
        learned_basepolicy=None,
    ):
        self.llm_tale = llm_tale
        self.rl_policy = self.llm_tale.rl_policy
        self.learned_basepolicy = self.llm_tale.learned_basepolicy
        self.llm_tale.set_env(
            task_objects=self.task_objects,
            _post_process_obs=self._post_process_obs,
            change_device=self.change_device,
            get_picked_obj=None,
            is_grasping=self.check_grasp,
            on_policy=self.on_policy,
        )

    def restart_simulation(self):
        raise NotImplementedError("You need to implement the restart_simulation method in your environment class.")

    def set_llm_plan(self, llm_plan_path):
        if self.LLM_instuction:
            self.llm_tale.set_llm_plan(llm_plan_path)

    def _process_obs(self, obs_dict):
        gripper_pose = torch.tensor(obs_dict.gripper_pose, dtype=torch.float32)
        q_vel = torch.tensor(obs_dict.joint_velocities, dtype=torch.float32)
        gripper_open = torch.tensor([obs_dict.gripper_open], dtype=torch.float32)
        goal_space = torch.zeros(
            9,
        )
        obs = {
            "gripper_open": gripper_open,
            "gripper_pose": gripper_pose,
            "q_vel": q_vel,
            "goal_space": goal_space,
        }
        if not self.image_input:
            obs["obj_states"] = self.get_object_state(obs_dict)
        else:
            raise NotImplementedError
        if self.LLM_instuction:
            start_choose = self.current_episode > self.llm_tale.start_choice_ep
            goal_space = self.llm_tale.step_high_level(obs, self.objects, start_choose, self.eval)
            obs = self.llm_tale.reform_obs_with_action(obs, goal_space)

        observation = self._post_process_obs(obs, goal_space, replace_goal_space=True)
        return observation

    def get_object_state(self, obs_dict):
        raise NotImplementedError

    def _post_process_obs(self, obs, goal_space, replace_goal_space=False):
        if replace_goal_space:
            obs["goal_space"] = goal_space
            self.observation = obs
        state = torch.cat([obs["gripper_pose"], obs["gripper_open"], obs["q_vel"], goal_space])
        if self.LLM_instuction:
            state = torch.cat([state, obs["base_action"]])
        if not self.image_input:
            state = torch.cat([state, obs["obj_states"]])
        else:
            observation = {"state": state, "image": obs["image"]}
            state = torch.cat([(observation["image"] / 255 - 0.5).flatten(), observation["state"]])
        observation = state.unsqueeze(0)
        return observation

    def _process_action(self, action):
        if action.device != "cpu":
            action = action.cpu()
        if self.pos_only:
            action = torch.cat([action[:3], torch.zeros(3), action[3:4]])
        if self.LLM_instuction:
            target_pose = self.observation["goal_space"]
            gripper_pose = self.observation["gripper_pose"]
            if self.learned_basepolicy is None:
                action = self.llm_tale._prim["residual_action"](
                    gripper_poses=gripper_pose,
                    target_poses=target_pose,
                    prim=self.llm_tale.plans[self.llm_tale.current_plannum],
                    action=action,
                )
            else:
                action = action + self.observation["base_action"]
            action = self.llm_tale.plans[self.llm_tale.current_plannum].prime_action(action)
        action[-1] = (action[-1] + 1) / 2
        return action

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self.step_num += 1
        actions = actions[0] if actions.ndim == 2 else actions
        raw_actions = actions
        actions = self._process_action(actions)
        res_action = np.zeros(
            8,
        )
        res_action[:3] = actions[:3].detach().numpy()
        res_action[-1] = actions[-1].detach().numpy()
        res_action[3:7] = axisangle2quat(actions[3:6]).detach().numpy()
        try:
            self._obs_dict, reward, terminated = self.task.step(res_action)
        except Exception as e:
            print("Error in RLBench step:", e)
            terminated = True
            reward = -10.0
            self._obs_dict = deepcopy(self.last_obs_dict)

        self.last_obs_dict = self._obs_dict
        obs = self._process_obs(self._obs_dict)
        obs = self.change_device(obs)

        # Base reward/terminated from RLBench (make Python types first)
        reward_py = float(reward)
        terminated_py = bool(terminated)  # RLBench often returns bool already
        truncated_py = self.step_num >= self.episode_length

        # Wrap as tensors on the right device (0-dim scalars)
        reward_t = torch.tensor(reward_py, device=self.device, dtype=torch.float32)
        terminated_t = torch.tensor(terminated_py, device=self.device, dtype=torch.bool)
        truncated_t = torch.tensor(truncated_py, device=self.device, dtype=torch.bool)

        # info should carry only Python scalars
        info = {"success": reward > 0.99}

        # Optional LLM_TALE reward (keep Python numeric, then tensorize)
        if self.LLM_instuction and not self.expert_reward:
            gripper_pose = self.observation["gripper_pose"]
            r_llm_tale = float(self.llm_tale.get_llm_tale_reward(gripper_pose, raw_actions, self.observation))
            reward_t = reward_t + torch.tensor(r_llm_tale, device=self.device, dtype=torch.float32)

        # Task-specific shaping (Python types in/out)
        if self.LLM_instuction:
            info = self.get_task_relevent_info(info)  # info contains Python bools/floats
            ts_reward_py, ts_terminated_py = self.task_relevant_reward(info)
            reward_t = reward_t + torch.tensor(float(ts_reward_py), device=self.device, dtype=torch.float32)
            terminated_t = torch.logical_or(terminated_t, torch.tensor(bool(ts_terminated_py), device=self.device))

        # Use Python booleans for control flow (avoid syncing on tensors)
        if bool(terminated_t.item()) or bool(truncated_t.item()):
            # merge success with terminated (Python side first)
            success = bool(info["success"])

            self.success_memory[self.current_episode] = success
            if self.rl_policy is not None:
                self.rl_policy.track_data("Episode /Success Rate", int(success))
                if self.LLM_instuction:
                    pick_success = 1 if self.llm_tale.plans[0].success else 0
                    self.rl_policy.track_data("Episode /Pick Success Rate", pick_success)

        # Final scaling (tensor)
        reward_t = reward_t * float(self.reward_scale)
        return obs, reward_t, terminated_t, truncated_t, info

    def check_grasp(self):
        return check_object_grasp(self.task)

    def change_device(self, obs):
        if isinstance(obs, dict):
            for key in obs.keys():
                obs[key] = obs[key].to(self.device)
        else:
            obs = obs.to(self.device)
        return obs

    def set_objects(self):
        pass

    def reset(self, fake_reset=False, seed=None, options=None) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        self.step_num = 0
        self.current_episode += 1
        _, self._obs_dict = self.task.reset()
        self.last_obs_dict = self._obs_dict
        self.set_objects()

        if fake_reset:
            return self._obs_dict, {}
        if self.LLM_instuction:
            self.llm_tale.reset_high_level()
        obs = self._process_obs(self._obs_dict)
        obs = self.change_device(obs)
        info = {}
        if self.LLM_instuction:
            info = self.init_task_relevent_info(info)
        return obs, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        if self._obs_dict is not None:
            frame = self._obs_dict.front_rgb
            if frame is not None:
                return frame

    def close(self) -> None:
        """Close the environment"""
        self._env.shutdown()
        self._closed = True

    def init_task_relevent_info(self, info):
        return info

    def get_task_relevent_info(self, info):
        return info

    def task_relevant_reward(self, info):
        if info["success"]:
            return 100, torch.tensor([True])
        else:
            return 0, torch.tensor([False])
